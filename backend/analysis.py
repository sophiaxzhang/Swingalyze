import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.distance import euclidean
from typing import List, Tuple, Dict, Optional
import json
import os
from pathlib import Path
import openai
import hashlib
from functools import lru_cache
import re
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#converts objects into JSON-serializable types 
def safe_convert(obj):
    if isinstance(obj, dict):
        return {k: safe_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_convert(x) for x in obj]
    elif isinstance(obj, (float, int)):
        return obj
    elif hasattr(obj, 'tolist'): #for numpy arrays
        return obj.tolist()
    return obj

class SwingCoach:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.cache = {}
    
    def _compute_swing_flags(self, analysis_data: Dict) -> Dict[str, any]:
        """Compute badminton swing flags and metrics for LLM analysis"""
        flags = {}
        
        angles = analysis_data.get('angles', {})
        swing_phases = analysis_data.get('swing_phases', {})
        similarity_scores = analysis_data.get('similarity_scores', {})
        
        if swing_phases:
            racket_prep_length = swing_phases.get('racket_prep', 0) - swing_phases.get('setup', 0)
            backswing_length = swing_phases.get('top_of_backswing', 0) - swing_phases.get('backswing_start', 0)
            downswing_length = swing_phases.get('contact', 0) - swing_phases.get('downswing_start', 0)
            follow_through_length = swing_phases.get('follow_through', 0) - swing_phases.get('contact', 0)
            
            flags['racket_prep_timing'] = {
                'frames': racket_prep_length,
                'issue': racket_prep_length < 2 or racket_prep_length > 8 #too rushed or too slow
            }
            
            flags['backswing_timing'] = {
                'frames': backswing_length,
                'issue': backswing_length < 3 or backswing_length > 12 #too fast or too slow
            }
            
            flags['downswing_timing'] = {
                'frames': downswing_length,
                'issue': downswing_length < 2 or downswing_length > 8
            }
            
            flags['follow_through_timing'] = {
                'frames': follow_through_length,
                'issue': follow_through_length < 2 or follow_through_length > 10 #incomplete follow through or  over
            }
        
        #angle-based flags
        if angles:
            #wrist angle at contact should show proper snap
            if 'wrist_angle' in angles and swing_phases.get('contact'):
                contact_frame = min(swing_phases['contact'], len(angles['wrist_angle']) - 1)
                wrist_at_contact = angles['wrist_angle'][contact_frame]
                flags['wrist_snap_contact'] = {
                    'value': wrist_at_contact,
                    'issue': wrist_at_contact < 60 or wrist_at_contact > 100 #bad pronation
                }
            
            #shoulder rotation at top of backswing
            if 'shoulder_rotation' in angles and swing_phases.get('top_of_backswing'):
                top_frame = min(swing_phases['top_of_backswing'], len(angles['shoulder_rotation']) - 1)
                shoulder_at_top = angles['shoulder_rotation'][top_frame]
                flags['shoulder_rotation_top'] = {
                    'value': shoulder_at_top,
                    'issue': shoulder_at_top < 90 or shoulder_at_top > 140 #bad rotation
                }
            
            #hip rotation at contact
            if 'hip_rotation' in angles and swing_phases.get('contact'):
                contact_frame = min(swing_phases['contact'], len(angles['hip_rotation']) - 1)
                hip_at_contact = angles['hip_rotation'][contact_frame]
                flags['hip_rotation_contact'] = {
                    'value': hip_at_contact,
                    'issue': hip_at_contact < 80 or hip_at_contact > 120 #bad hip pos
                }
            
            #racket prep angle
            if 'racket_prep_angle' in angles and swing_phases.get('racket_prep'):
                prep_frame = min(swing_phases['racket_prep'], len(angles['racket_prep_angle']) - 1)
                racket_at_prep = angles['racket_prep_angle'][prep_frame]
                flags['racket_prep_position'] = {
                    'value': racket_at_prep,
                    'issue': racket_at_prep < 70 or racket_at_prep > 110  #bad racket angle
                }
        
        #similarity score flags
        if similarity_scores:
            for angle_name, score in similarity_scores.items():
                flags[f'{angle_name}_similarity'] = {
                    'score': score,
                    'issue': score < 0.7 #bad similarity to reference
                }
        
        #contact heigh analysis
        if 'contact_height' in analysis_data:
            contact_height = analysis_data['contact_height']
            flags['contact_height'] = {
                'value': contact_height,
                'issue': contact_height < 0.3 or contact_height > 0.8  #too low or high
            }
        
        return flags
    
    def _create_llm_prompt(self, phases, angles, similarity_scores, swing_flags):
        return f"""
    Analyze the badminton swing data and return STRICT JSON ONLY. Do NOT include extra text or markdown.

    Here is the swing analysis data:
    - Similarity Scores: {json.dumps(safe_convert(similarity_scores))}
    - Phases Detected: {json.dumps(safe_convert(phases))}
    - Angles: {json.dumps(safe_convert(angles))}
    - Swing Flags: {json.dumps(safe_convert(swing_flags))}

    Your task:
    Provide an actionable analysis for the player, focusing on what needs improvement and what they did well.

    JSON SCHEMA:
    {{
    "summary": "Brief overall feedback (1-2 sentences)",
    "priority_issues": ["Top 3-5 problems in the swing"],
    "specific_tips": ["Practical, actionable tips for fixing each issue"],
    "positive_feedback": ["Things they did well to encourage them"],
    "drill_suggestions": ["Drills or exercises to practice improvement"]
    }}

    Return ONLY valid JSON, no explanations, no markdown, no extra text.
    """

    def get_swing_advice(self, phases, angles, similarity_scores, key_metrics, reference_metrics):
        try:
            analysis_data = {
                "similarity_scores": safe_convert(similarity_scores),
                "phases": safe_convert(phases),
                "angles": safe_convert(angles),
                "key_metrics": safe_convert(key_metrics),
                "reference_metrics": safe_convert(reference_metrics)
            }

            #compute cache_key with json-safe hashing
            cache_key = hashlib.md5(json.dumps(analysis_data, sort_keys=True).encode()).hexdigest()
            cached = self.get_cached_advice(cache_key)
            if cached:
                logger.info("Returning cached advice")
                return cached

            #comput swing flags
            analysis_data = {
                "phases": phases,
                "angles": angles,
                "similarity_scores": similarity_scores,
                "key_metrics": key_metrics,
                "reference_metrics": reference_metrics
            }

            swing_flags = self._compute_swing_flags(analysis_data)

            prompt = self._create_llm_prompt(phases, angles, similarity_scores, swing_flags)

            #call LLM with retry logic (up to 3 trys)
            response_text = None
            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a professional badminton coach. Return ONLY valid JSON with analysis."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=1000
                    )
                    response_text = response.choices[0].message.content
                    break
                except Exception as e:
                    logger.warning(f"LLM request failed (attempt {attempt+1}): {e}")
                    time.sleep(2)

            if not response_text:
                raise ValueError("No response from LLM")

            logger.debug(f"Raw LLM response: {response_text}")

            #extract JSON using regex to handle  extra text
            match = re.search(r"\{.*\}", response_text, re.DOTALL) #ignore extra text
            if not match:
                raise ValueError("No JSON found in LLM response")
            advice_json = json.loads(match.group())

            #cache the result
            self.cache[cache_key] = advice_json
            return advice_json

        except Exception as e:
            logger.error(f"Error in get_swing_advice: {e}")

            #use swing_flags if available
            fallback_advice = {
                "summary": "Basic analysis completed",
                "priority_issues": swing_flags.get("issues", ["Could not retrieve detailed advice"]),
                "specific_tips": ["Work on consistency and having a smooth swing"],
                "positive_feedback": ["Keep practicing! Your form shows potential."],
                "drill_suggestions": ["Shadow swings", "Slow-motion swings to practice the different phases"]
            }
            return fallback_advice
    
    #clear advice cache
    def clear_cache(self):
        self.cache.clear()
    
    #get cahed advice by key
    def get_cached_advice(self, cache_key: str) -> Optional[Dict]:
        return self.cache.get(cache_key)

class SwingAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        #MediaPipe pose landmark indices
        self.landmark_indices = {
            'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
        
    #use frame data to extract post keypoints from video
    def extract_keypoints(self, video_path: str) -> List[Dict]:
        cap = cv2.VideoCapture(video_path)
        keypoints_data = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            #convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            frame_data = {
                'frame': frame_count,
                'landmarks': None,
                'timestamp': frame_count / fps,
                'confidence': 0.0
            }
            
            if results.pose_landmarks:
                landmarks = []
                total_confidence = 0
                valid_landmarks = 0
                
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                    if landmark.visibility > 0.5:
                        total_confidence += landmark.visibility
                        valid_landmarks += 1
                
                frame_data['landmarks'] = landmarks
                frame_data['confidence'] = total_confidence / max(valid_landmarks, 1)
            
            keypoints_data.append(frame_data)
            frame_count += 1
            
        cap.release()
        return keypoints_data
    
    #calculate angle between three points (p2 is the vertex)
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  #handle numerical errors
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    #use metrics to extract key angles
    def extract_key_angles(self, keypoints_data: List[Dict]) -> Dict[str, List[float]]:
        
        angles = {
            'left_elbow': [],
            'right_elbow': [],
            'left_shoulder': [],
            'right_shoulder': [],
            'left_hip': [],
            'right_hip': [],
            'left_knee': [],
            'right_knee': [],
            'spine_angle': [],
            'wrist_angle': [], #wrist pronation
            'ankle_angle': [],
            'shoulder_rotation': [],
            'hip_rotation': [],
            'racket_prep_angle': [],
        }
        
        for frame_data in keypoints_data:
            landmarks = frame_data.get('landmarks')
            if not landmarks:
                #fill with NaN if no landmarks detected
                for angle_name in angles:
                    angles[angle_name].append(float('nan'))
                continue
            
            #get keypoints for angle calculations
            left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
            right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
            left_elbow = (landmarks[13]['x'], landmarks[13]['y'])
            right_elbow = (landmarks[14]['x'], landmarks[14]['y'])
            left_wrist = (landmarks[15]['x'], landmarks[15]['y'])
            right_wrist = (landmarks[16]['x'], landmarks[16]['y'])
            left_hip = (landmarks[23]['x'], landmarks[23]['y'])
            right_hip = (landmarks[24]['x'], landmarks[24]['y'])
            left_knee = (landmarks[25]['x'], landmarks[25]['y'])
            right_knee = (landmarks[26]['x'], landmarks[26]['y'])
            left_ankle = (landmarks[27]['x'], landmarks[27]['y'])
            right_ankle = (landmarks[28]['x'], landmarks[28]['y'])
            
            #calculate basic angles
            left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_shoulder_angle = self.calculate_angle(left_elbow, left_shoulder, left_hip)
            right_shoulder_angle = self.calculate_angle(right_elbow, right_shoulder, right_hip)
            left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            #spine angle (angle between shoulders and hips)
            spine_angle = self.calculate_angle(
                ((left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2),
                ((left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2),
                (0, 1) #vertical reference
            )
            
            #wrist pronation angle
            wrist_angle = self.calculate_angle(right_elbow, right_wrist, (right_wrist[0] + 0.1, right_wrist[1]))
            
            ankle_angle = self.calculate_angle(left_knee, left_ankle, (left_ankle[0], left_ankle[1] + 0.1))
            
            #shoulder rotation (angle between shoulder line and horizontal)
            shoulder_rotation = self.calculate_angle(
                left_shoulder,
                right_shoulder,
                (right_shoulder[0] + 0.1, right_shoulder[1])
            )
            
            #hip rotation (angle between hip line and horizontal)
            hip_rotation = self.calculate_angle(
                left_hip,
                right_hip,
                (right_hip[0] + 0.1, right_hip[1])
            )
            
            #racket preparation angle which represents the angle of the racket arm relative to the body
            racket_prep_angle = self.calculate_angle(
                right_shoulder,
                right_elbow,
                (right_elbow[0], right_elbow[1] - 0.1)  # Vertical reference
            )
            
            #store all angles
            angles['left_elbow'].append(left_elbow_angle)
            angles['right_elbow'].append(right_elbow_angle)
            angles['left_shoulder'].append(left_shoulder_angle)
            angles['right_shoulder'].append(right_shoulder_angle)
            angles['left_hip'].append(left_hip_angle)
            angles['right_hip'].append(right_hip_angle)
            angles['left_knee'].append(left_knee_angle)
            angles['right_knee'].append(right_knee_angle)
            angles['spine_angle'].append(spine_angle)
            angles['wrist_angle'].append(wrist_angle)
            angles['ankle_angle'].append(ankle_angle)
            angles['shoulder_rotation'].append(shoulder_rotation)
            angles['hip_rotation'].append(hip_rotation)
            angles['racket_prep_angle'].append(racket_prep_angle)
        
        return angles
    
    #use multiple signals to detect swing phases
    def detect_swing_phases(self, angles: Dict[str, List[float]]) -> Dict[str, int]:
        signals = []
        
        #right elbow velocity (main swing indicator)
        if 'right_elbow' in angles:
            elbow_angles = np.array(angles['right_elbow'])
            elbow_velocity = np.gradient(elbow_angles)
            signals.append(elbow_velocity)
        
        #shoulder rotation velocity
        if 'shoulder_rotation' in angles:
            shoulder_angles = np.array(angles['shoulder_rotation'])
            shoulder_velocity = np.gradient(shoulder_angles)
            signals.append(shoulder_velocity)
        
        #spine angle changes
        if 'spine_angle' in angles:
            spine_angles = np.array(angles['spine_angle'])
            spine_changes = np.gradient(spine_angles)
            signals.append(spine_changes)
        
        #wrist angle changes
        if 'wrist_angle' in angles:
            wrist_angles = np.array(angles['wrist_angle'])
            wrist_changes = np.gradient(wrist_angles)
            signals.append(wrist_changes)
        
        if not signals:
            return {}
        
        #cCombine signals for robust detection
        combined_signal = np.mean(signals, axis=0)
        
        #smooth the signal
        if len(combined_signal) > 10:
            combined_signal = savgol_filter(combined_signal, min(11, len(combined_signal) // 2), 3)
        
        #find peaks in the signal
        peaks, _ = find_peaks(np.abs(combined_signal), height=np.std(combined_signal) * 0.3, distance=3)
        
        if len(peaks) < 2:
            #fallback: create basic phases if not enough peaks detected
            return {
                'setup': 0,
                'racket_prep': len(combined_signal) // 4,
                'backswing_start': len(combined_signal) // 3,
                'top_of_backswing': len(combined_signal) // 2,
                'downswing_start': len(combined_signal) // 2,
                'contact': 3 * len(combined_signal) // 4,
                'follow_through': len(combined_signal) - 1
            }
        
        #sort peaks by signal magnitude
        peak_magnitudes = np.abs(combined_signal[peaks])
        sorted_peaks = peaks[np.argsort(peak_magnitudes)[::-1]]
        
        #define badminton swing phases
        phases = {}
        
        #setup phase (start of video)
        phases['setup'] = 0
        
        #racket prepn phase (before backswing)
        if len(sorted_peaks) >= 1:
            prep_peak = sorted_peaks[0]
            phases['racket_prep'] = max(0, prep_peak - 2)
        else:
            phases['racket_prep'] = len(combined_signal) // 4
        
        #backswing start (first significant movement)
        if len(sorted_peaks) >= 2:
            backswing_peak = sorted_peaks[1]
            phases['backswing_start'] = max(0, backswing_peak - 1)
        else:
            phases['backswing_start'] = len(combined_signal) // 3
        
        #top of backswing (maximum backswing)
        if len(sorted_peaks) >= 3:
            top_peak = sorted_peaks[2]
            phases['top_of_backswing'] = top_peak
        else:
            phases['top_of_backswing'] = len(combined_signal) // 2
        
        #downswing start (beginning of forward motion)
        if 'top_of_backswing' in phases:
            top_frame = phases['top_of_backswing']
            #look for the start of downswing after top
            for i in range(top_frame + 1, min(top_frame + 10, len(combined_signal))):
                if combined_signal[i] > 0:  #pos velocity indicates downswing
                    phases['downswing_start'] = i
                    break
            #fallback if downswing not detected
            if 'downswing_start' not in phases:
                phases['downswing_start'] = min(top_frame + 5, len(combined_signal) - 1)
        
        #contact point with birdie
        if 'downswing_start' in phases:
            downswing_start = phases['downswing_start']
            #look for the contact point (peak in wrist movement)
            if 'wrist_angle' in angles:
                wrist_angles = np.array(angles['wrist_angle'])
                contact_window = wrist_angles[downswing_start:min(downswing_start + 15, len(wrist_angles))]
                if len(contact_window) > 0:
                    contact_peak = np.argmax(np.abs(np.gradient(contact_window))) + downswing_start
                    phases['contact'] = contact_peak
                else:
                    phases['contact'] = min(downswing_start + 8, len(combined_signal) - 1)
            else:
                phases['contact'] = min(downswing_start + 8, len(combined_signal) - 1)
        
        #follow through
        if 'contact' in phases:
            contact_frame = phases['contact']
            #follow through usualy 5-10 frames after contact
            follow_through_frame = min(contact_frame + 8, len(combined_signal) - 1)
            phases['follow_through'] = follow_through_frame
        else:
            phases['follow_through'] = len(combined_signal) - 1
        
        return phases
    
    #normalize swing timing to starndard num of frames
    def normalize_swing_timing(self, angles: Dict[str, List[float]], 
                             target_frames: int = 100) -> Dict[str, List[float]]:
        normalized = {}
        
        for key, values in angles.items():
            if key == 'timestamps':
                normalized[key] = np.linspace(0, 1, target_frames)
            else:
                values_array = np.array(values)
                if len(values_array) == 0:
                    normalized[key] = np.full(target_frames, np.nan)
                else:
                    #interpolate to target length
                    original_indices = np.linspace(0, len(values_array)-1, len(values_array))
                    target_indices = np.linspace(0, len(values_array)-1, target_frames)
                    normalized[key] = np.interp(target_indices, original_indices, values_array)
        
        return normalized
    
    #works no matter when swing occurs in video
    def align_swings_robust(self, swings_data: List[Dict[str, List[float]]], 
                          reference_swing: Optional[Dict[str, List[float]]] = None) -> List[Dict[str, List[float]]]:
        aligned_swings = []
        
        if reference_swing is None:
            reference_swing = swings_data[0] if swings_data else None
        
        if reference_swing is None:
            return aligned_swings
        
        reference_phases = self.detect_swing_phases(reference_swing)
        
        #check if reference phases are complete
        required_phases = ['backswing_start', 'follow_through']
        if not all(phase in reference_phases for phase in required_phases):
            print(f"Warning: Reference swing missing required phases: {required_phases}")
            return aligned_swings
        
        reference_length = reference_phases['follow_through'] - reference_phases['backswing_start']
        
        for swing in swings_data:
            try:
                #detect phases for current swing
                phases = self.detect_swing_phases(swing)
                
                #check if current swing has all required phases
                if not all(phase in phases for phase in required_phases):
                    print(f"Warning: Swing missing required phases: {required_phases}")
                    continue
                
                swing_start = phases['backswing_start']
                swing_end = phases['follow_through']
                
                aligned_swing = {}
                
                for key, values in swing.items():
                    if key == 'timestamps':
                        #create normalized timeline
                        aligned_swing[key] = np.linspace(0, 1, reference_length)
                    else:
                        #extract swing portion and normalize
                        swing_portion = np.array(values[swing_start:swing_end+1])
                        if len(swing_portion) == 0:
                            aligned_swing[key] = np.full(reference_length, np.nan)
                        else:
                            #interpolate to reference length
                            original_indices = np.linspace(0, len(swing_portion)-1, len(swing_portion))
                            target_indices = np.linspace(0, len(swing_portion)-1, reference_length)
                            aligned_swing[key] = np.interp(target_indices, original_indices, swing_portion)
                
                aligned_swings.append(aligned_swing)
                
            except Exception as e:
                print(f"Error aligning swing: {e}")
                continue
        
        return aligned_swings
    
    def calculate_contact_height(self, keypoints_data: List[Dict], swing_phases: Dict[str, int]) -> float:
        if 'contact' not in swing_phases:
            return 0.5  #default middle height if contact not detected
        
        contact_frame = swing_phases['contact']
        if contact_frame >= len(keypoints_data):
            return 0.5
        
        #get wrist position at contact (rackets held at wrist level)
        contact_data = keypoints_data[contact_frame]
        if not contact_data.get('landmarks'):
            return 0.5
        
        #use right wrist height (normalized 0-1, where 0 is top of frame, 1 is bottom)
        right_wrist = contact_data['landmarks'][16]  #right wrist index
        contact_height = right_wrist['y']  #y is height in MediaPipe (0=top, 1=bottom)
        
        return contact_height
    
    #analysis pipeline w/ frame by frame data
    def analyze_video_detailed(self, video_path: str) -> Dict:
        print(f"Extracting keypoints from {video_path}...")
        keypoints = self.extract_keypoints(video_path)
        
        print("Calculating angles...")
        angles = self.extract_key_angles(keypoints)
        
        print("Detecting swing phases...")
        phases = self.detect_swing_phases(angles)
        
        print("Calculating contact height...")
        contact_height = self.calculate_contact_height(keypoints, phases)
        
        print("Normalizing timing...")
        normalized_angles = self.normalize_swing_timing(angles)
        
        return {
            'keypoints': keypoints,
            'angles': angles,
            'normalized_angles': normalized_angles,
            'swing_phases': phases,
            'video_path': video_path,
            'frame_count': len(keypoints),
            'fps': len(keypoints) / max(keypoints[-1]['timestamp'], 1) if keypoints else 0,
            'contact_height': contact_height
        }
    
    def compare_swings_robust(self, video_paths: List[str], 
                            reference_video: Optional[str] = None) -> Dict:
        all_swings = []
        reference_swing = None
        
        for i, video_path in enumerate(video_paths):
            swing_data = self.analyze_video_detailed(video_path)
            all_swings.append(swing_data)
            
            #set reference swing
            if reference_video and video_path == reference_video:
                reference_swing = swing_data['angles']
            elif reference_swing is None and i == 0:
                reference_swing = swing_data['angles']
        
        print("Aligning swings robustly...")
        aligned_swings = self.align_swings_robust([s['angles'] for s in all_swings], reference_swing)
        
        return {
            'original_swings': all_swings,
            'aligned_swings': aligned_swings,
            'video_paths': video_paths,
            'reference_video': reference_video
        }
    
    #save ref data for video
    def save_reference_data(self, video_path: str, output_path: str = "reference_data.json"):
        analysis = self.analyze_video_detailed(video_path)
        
        #convert numpy arrays to lists for JSON serialization
        serializable_analysis = {}
        for key, value in analysis.items():
            if isinstance(value, dict):
                serializable_analysis[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        serializable_analysis[key][sub_key] = sub_value.tolist()
                    else:
                        serializable_analysis[key][sub_key] = sub_value
            elif isinstance(value, np.ndarray):
                serializable_analysis[key] = value.tolist()
            else:
                serializable_analysis[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)
        
        print(f"Reference data saved to {output_path}")
        return serializable_analysis
    
    def load_reference_data(self, reference_path: str) -> Dict:
        with open(reference_path, 'r') as f:
            return json.load(f)
    
    #plot alignedswing comparisons
    def plot_comparison(self, comparison_data: Dict, angle_name: str = 'right_elbow'):
        aligned_swings = comparison_data['aligned_swings']
        
        plt.figure(figsize=(12, 8))
        
        for i, swing in enumerate(aligned_swings):
            plt.plot(swing['timestamps'], swing[angle_name], 
                    label=f'Swing {i+1}', linewidth=2, alpha=0.7)
        
        plt.xlabel('Normalized Time')
        plt.ylabel(f'{angle_name.replace("_", " ").title()} Angle (degrees)')
        plt.title(f'{angle_name.replace("_", " ").title()} Angle Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    analyzer = SwingAnalyzer()