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
        
        # MediaPipe pose landmark indices
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
        
    def extract_keypoints(self, video_path: str) -> List[Dict]:
        """Extract pose keypoints from video with enhanced frame data"""
        cap = cv2.VideoCapture(video_path)
        keypoints_data = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
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
    
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """Calculate angle between three points (p2 is the vertex)"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def extract_key_angles(self, keypoints_data: List[Dict]) -> Dict[str, List[float]]:
        """Extract key angles for golf swing analysis with enhanced measurements"""
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
            'wrist_angle': [],
            'ankle_angle': [],
            'shoulder_rotation': [],
            'hip_rotation': [],
            'timestamps': []
        }
        
        for frame_data in keypoints_data:
            if frame_data['landmarks'] is None:
                # Fill with NaN for missing frames
                for key in angles.keys():
                    if key != 'timestamps':
                        angles[key].append(np.nan)
                angles['timestamps'].append(frame_data['timestamp'])
                continue
                
            landmarks = frame_data['landmarks']
            
            def get_point(idx):
                return (landmarks[idx]['x'], landmarks[idx]['y'])
            
            try:
                # Basic joint angles
                left_elbow_angle = self.calculate_angle(
                    get_point(self.landmark_indices['left_shoulder']),
                    get_point(self.landmark_indices['left_elbow']),
                    get_point(self.landmark_indices['left_wrist'])
                )
                angles['left_elbow'].append(left_elbow_angle)
                
                right_elbow_angle = self.calculate_angle(
                    get_point(self.landmark_indices['right_shoulder']),
                    get_point(self.landmark_indices['right_elbow']),
                    get_point(self.landmark_indices['right_wrist'])
                )
                angles['right_elbow'].append(right_elbow_angle)
                
                # Shoulder angles
                left_shoulder_angle = self.calculate_angle(
                    get_point(self.landmark_indices['left_elbow']),
                    get_point(self.landmark_indices['left_shoulder']),
                    get_point(self.landmark_indices['left_hip'])
                )
                angles['left_shoulder'].append(left_shoulder_angle)
                
                right_shoulder_angle = self.calculate_angle(
                    get_point(self.landmark_indices['right_elbow']),
                    get_point(self.landmark_indices['right_shoulder']),
                    get_point(self.landmark_indices['right_hip'])
                )
                angles['right_shoulder'].append(right_shoulder_angle)
                
                # Hip angles
                left_hip_angle = self.calculate_angle(
                    get_point(self.landmark_indices['left_shoulder']),
                    get_point(self.landmark_indices['left_hip']),
                    get_point(self.landmark_indices['left_knee'])
                )
                angles['left_hip'].append(left_hip_angle)
                
                right_hip_angle = self.calculate_angle(
                    get_point(self.landmark_indices['right_shoulder']),
                    get_point(self.landmark_indices['right_hip']),
                    get_point(self.landmark_indices['right_knee'])
                )
                angles['right_hip'].append(right_hip_angle)
                
                # Knee angles
                left_knee_angle = self.calculate_angle(
                    get_point(self.landmark_indices['left_hip']),
                    get_point(self.landmark_indices['left_knee']),
                    get_point(self.landmark_indices['left_ankle'])
                )
                angles['left_knee'].append(left_knee_angle)
                
                right_knee_angle = self.calculate_angle(
                    get_point(self.landmark_indices['right_hip']),
                    get_point(self.landmark_indices['right_knee']),
                    get_point(self.landmark_indices['right_ankle'])
                )
                angles['right_knee'].append(right_knee_angle)
                
                # Wrist angle (elbow-wrist-finger)
                left_wrist_angle = self.calculate_angle(
                    get_point(self.landmark_indices['left_elbow']),
                    get_point(self.landmark_indices['left_wrist']),
                    get_point(self.landmark_indices['left_index'])
                )
                angles['wrist_angle'].append(left_wrist_angle)
                
                # Ankle angle (knee-ankle-toe)
                left_ankle_angle = self.calculate_angle(
                    get_point(self.landmark_indices['left_knee']),
                    get_point(self.landmark_indices['left_ankle']),
                    get_point(self.landmark_indices['left_foot_index'])
                )
                angles['ankle_angle'].append(left_ankle_angle)
                
                # Spine angle
                shoulder_midpoint = (
                    (get_point(self.landmark_indices['left_shoulder'])[0] + get_point(self.landmark_indices['right_shoulder'])[0]) / 2,
                    (get_point(self.landmark_indices['left_shoulder'])[1] + get_point(self.landmark_indices['right_shoulder'])[1]) / 2
                )
                hip_midpoint = (
                    (get_point(self.landmark_indices['left_hip'])[0] + get_point(self.landmark_indices['right_hip'])[0]) / 2,
                    (get_point(self.landmark_indices['left_hip'])[1] + get_point(self.landmark_indices['right_hip'])[1]) / 2
                )
                
                spine_vector = np.array([shoulder_midpoint[0] - hip_midpoint[0], 
                                       shoulder_midpoint[1] - hip_midpoint[1]])
                vertical_vector = np.array([0, -1])
                
                cos_spine_angle = np.dot(spine_vector, vertical_vector) / (np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector))
                cos_spine_angle = np.clip(cos_spine_angle, -1.0, 1.0)
                spine_angle = np.degrees(np.arccos(cos_spine_angle))
                angles['spine_angle'].append(spine_angle)
                
                # Shoulder rotation (angle between shoulders and horizontal)
                shoulder_vector = np.array([
                    get_point(self.landmark_indices['right_shoulder'])[0] - get_point(self.landmark_indices['left_shoulder'])[0],
                    get_point(self.landmark_indices['right_shoulder'])[1] - get_point(self.landmark_indices['left_shoulder'])[1]
                ])
                horizontal_vector = np.array([1, 0])
                
                cos_shoulder_rotation = np.dot(shoulder_vector, horizontal_vector) / (np.linalg.norm(shoulder_vector) * np.linalg.norm(horizontal_vector))
                cos_shoulder_rotation = np.clip(cos_shoulder_rotation, -1.0, 1.0)
                shoulder_rotation = np.degrees(np.arccos(cos_shoulder_rotation))
                angles['shoulder_rotation'].append(shoulder_rotation)
                
                # Hip rotation
                hip_vector = np.array([
                    get_point(self.landmark_indices['right_hip'])[0] - get_point(self.landmark_indices['left_hip'])[0],
                    get_point(self.landmark_indices['right_hip'])[1] - get_point(self.landmark_indices['left_hip'])[1]
                ])
                
                cos_hip_rotation = np.dot(hip_vector, horizontal_vector) / (np.linalg.norm(hip_vector) * np.linalg.norm(horizontal_vector))
                cos_hip_rotation = np.clip(cos_hip_rotation, -1.0, 1.0)
                hip_rotation = np.degrees(np.arccos(cos_hip_rotation))
                angles['hip_rotation'].append(hip_rotation)
                
            except (IndexError, ZeroDivisionError, ValueError):
                # Handle cases where landmarks are missing or calculation fails
                for key in angles.keys():
                    if key != 'timestamps':
                        angles[key].append(np.nan)
            
            angles['timestamps'].append(frame_data['timestamp'])
        
        return angles
    
    def detect_swing_phases(self, angles: Dict[str, List[float]]) -> Dict[str, int]:
        """Detect different phases of the golf swing using multiple signals"""
        # Use multiple signals for robust swing detection
        signals = []
        
        # 1. Right elbow angle velocity
        right_elbow = np.array(angles['right_elbow'])
        right_elbow_velocity = np.gradient(right_elbow)
        right_elbow_velocity = np.nan_to_num(right_elbow_velocity)
        signals.append(right_elbow_velocity)
        
        # 2. Shoulder rotation velocity
        shoulder_rotation = np.array(angles['shoulder_rotation'])
        shoulder_velocity = np.gradient(shoulder_rotation)
        shoulder_velocity = np.nan_to_num(shoulder_velocity)
        signals.append(shoulder_velocity)
        
        # 3. Spine angle velocity
        spine_angle = np.array(angles['spine_angle'])
        spine_velocity = np.gradient(spine_angle)
        spine_velocity = np.nan_to_num(spine_velocity)
        signals.append(spine_velocity)
        
        # Combine signals for robust detection
        combined_signal = np.mean(signals, axis=0)
        
        # Apply smoothing
        if len(combined_signal) > 5:
            combined_signal = savgol_filter(combined_signal, min(5, len(combined_signal)//2), 2)
        
        # Find peaks (significant movements)
        peaks, _ = find_peaks(np.abs(combined_signal), height=np.std(combined_signal) * 1.5)
        
        if len(peaks) == 0:
            # Fallback: find first significant movement
            threshold = np.std(combined_signal) * 2
            significant_movement = np.where(np.abs(combined_signal) > threshold)[0]
            if len(significant_movement) > 0:
                return {
                    'setup': 0,
                    'backswing_start': significant_movement[0],
                    'top_of_backswing': min(significant_movement[0] + len(combined_signal)//4, len(combined_signal)-1),
                    'downswing_start': min(significant_movement[0] + len(combined_signal)//2, len(combined_signal)-1),
                    'impact': min(significant_movement[0] + 3*len(combined_signal)//4, len(combined_signal)-1),
                    'follow_through': len(combined_signal)-1
                }
            else:
                return {
                    'setup': 0,
                    'backswing_start': 0,
                    'top_of_backswing': len(combined_signal)//4,
                    'downswing_start': len(combined_signal)//2,
                    'impact': 3*len(combined_signal)//4,
                    'follow_through': len(combined_signal)-1
                }
        
        # Use peaks to identify swing phases
        peaks = sorted(peaks)
        
        return {
            'setup': 0,
            'backswing_start': peaks[0] if len(peaks) > 0 else 0,
            'top_of_backswing': peaks[1] if len(peaks) > 1 else len(combined_signal)//4,
            'downswing_start': peaks[2] if len(peaks) > 2 else len(combined_signal)//2,
            'impact': peaks[3] if len(peaks) > 3 else 3*len(combined_signal)//4,
            'follow_through': len(combined_signal)-1
        }
    
    def normalize_swing_timing(self, angles: Dict[str, List[float]], 
                             target_frames: int = 100) -> Dict[str, List[float]]:
        """Normalize swing timing to a standard number of frames"""
        normalized = {}
        
        for key, values in angles.items():
            if key == 'timestamps':
                normalized[key] = np.linspace(0, 1, target_frames)
            else:
                values_array = np.array(values)
                if len(values_array) == 0:
                    normalized[key] = np.full(target_frames, np.nan)
                else:
                    # Interpolate to target length
                    original_indices = np.linspace(0, len(values_array)-1, len(values_array))
                    target_indices = np.linspace(0, len(values_array)-1, target_frames)
                    normalized[key] = np.interp(target_indices, original_indices, values_array)
        
        return normalized
    
    def align_swings_robust(self, swings_data: List[Dict[str, List[float]]], 
                          reference_swing: Optional[Dict[str, List[float]]] = None) -> List[Dict[str, List[float]]]:
        """Robust swing alignment that works regardless of when swing occurs in video"""
        aligned_swings = []
        
        # Use reference swing if provided, otherwise use the first swing
        if reference_swing is None:
            reference_swing = swings_data[0] if swings_data else None
        
        if reference_swing is None:
            return aligned_swings
        
        # Detect phases for reference swing
        reference_phases = self.detect_swing_phases(reference_swing)
        reference_length = reference_phases['follow_through'] - reference_phases['backswing_start']
        
        for swing in swings_data:
            # Detect phases for current swing
            phases = self.detect_swing_phases(swing)
            
            # Extract swing portion (from backswing start to follow through)
            swing_start = phases['backswing_start']
            swing_end = phases['follow_through']
            
            aligned_swing = {}
            
            for key, values in swing.items():
                if key == 'timestamps':
                    # Create normalized timeline
                    aligned_swing[key] = np.linspace(0, 1, reference_length)
                else:
                    # Extract swing portion and normalize
                    swing_portion = np.array(values[swing_start:swing_end+1])
                    if len(swing_portion) == 0:
                        aligned_swing[key] = np.full(reference_length, np.nan)
                    else:
                        # Interpolate to reference length
                        original_indices = np.linspace(0, len(swing_portion)-1, len(swing_portion))
                        target_indices = np.linspace(0, len(swing_portion)-1, reference_length)
                        aligned_swing[key] = np.interp(target_indices, original_indices, swing_portion)
            
            aligned_swings.append(aligned_swing)
        
        return aligned_swings
    
    def analyze_video_detailed(self, video_path: str) -> Dict:
        """Complete analysis pipeline with detailed frame-by-frame data"""
        print(f"Extracting keypoints from {video_path}...")
        keypoints = self.extract_keypoints(video_path)
        
        print("Calculating angles...")
        angles = self.extract_key_angles(keypoints)
        
        print("Detecting swing phases...")
        phases = self.detect_swing_phases(angles)
        
        print("Normalizing timing...")
        normalized_angles = self.normalize_swing_timing(angles)
        
        return {
            'keypoints': keypoints,
            'angles': angles,
            'normalized_angles': normalized_angles,
            'swing_phases': phases,
            'video_path': video_path,
            'frame_count': len(keypoints),
            'fps': len(keypoints) / max(keypoints[-1]['timestamp'], 1) if keypoints else 0
        }
    
    def compare_swings_robust(self, video_paths: List[str], 
                            reference_video: Optional[str] = None) -> Dict:
        """Compare multiple swing videos with robust alignment"""
        all_swings = []
        reference_swing = None
        
        for i, video_path in enumerate(video_paths):
            swing_data = self.analyze_video_detailed(video_path)
            all_swings.append(swing_data)
            
            # Set reference swing
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
    
    def save_reference_data(self, video_path: str, output_path: str = "reference_data.json"):
        """Save detailed reference data for a video"""
        analysis = self.analyze_video_detailed(video_path)
        
        # Convert numpy arrays to lists for JSON serialization
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
        """Load reference data from file"""
        with open(reference_path, 'r') as f:
            return json.load(f)
    
    def plot_comparison(self, comparison_data: Dict, angle_name: str = 'right_elbow'):
        """Plot aligned swing comparisons"""
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

# Example usage
if __name__ == "__main__":
    analyzer = SwingAnalyzer()
    
    # Analyze single video with detailed data
    # swing_data = analyzer.analyze_video_detailed("golf_swing1.mp4")
    
    # Save reference data
    # analyzer.save_reference_data("reference.MOV", "reference_data.json")
    
    # Compare multiple videos with robust alignment
    # video_paths = ["swing1.mp4", "swing2.mp4", "swing3.mp4"]
    # comparison = analyzer.compare_swings_robust(video_paths, reference_video="reference.MOV")
    # analyzer.plot_comparison(comparison, 'right_elbow')
    
    print("Enhanced SwingAnalyzer ready! Use analyzer.analyze_video_detailed() or analyzer.compare_swings_robust()")