import cv2
import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent.resolve()
video_path = BASE_DIR / "reference.MOV"

data_path = BASE_DIR / "reference_analysis_detailed.json"

output_path = BASE_DIR / "swing_overlay_output.avi"

#load analysis data
try:
    with open(data_path) as f:
        swing_data = json.load(f)
    print("JSON loaded successfully!")
    print(f"Keys in swing_data: {list(swing_data.keys())}")
except Exception as e:
    print(f"Error loading JSON: {e}")
    exit(1)

#debug: Print structure
print("\nData structure analysis:")
if 'keypoints' in swing_data:
    keypoints_data = swing_data['keypoints']
    print(f"Keypoints type: {type(keypoints_data)}")
    if isinstance(keypoints_data, list) and len(keypoints_data) > 0:
        first_frame = keypoints_data[0]
        print(f"First keypoint item keys: {list(first_frame.keys()) if first_frame else 'Empty'}")
        if first_frame and 'landmarks' in first_frame and first_frame['landmarks']:
            sample_landmark = first_frame['landmarks'][0]
            print(f"Sample landmark: {sample_landmark}")
            print(f"Sample landmark keys: {list(sample_landmark.keys())}")
    elif isinstance(keypoints_data, dict):
        print(f"Keypoints dict keys: {list(keypoints_data.keys())}")

if 'angles' in swing_data:
    angles_data = swing_data['angles']
    print(f"Angles type: {type(angles_data)}")
    if isinstance(angles_data, dict):
        print(f"Angles dict keys: {list(angles_data.keys())}")

# convert data to expected format
keypoints_per_frame = []
angles_per_frame = []

#handle the detailed format from your test script
if 'keypoints' in swing_data and isinstance(swing_data['keypoints'], list):
    print(f"\nProcessing {len(swing_data['keypoints'])} frames...")
    
    for i, frame_data in enumerate(swing_data['keypoints']):
        frame_keypoints = {}
        
        if frame_data and 'landmarks' in frame_data and frame_data['landmarks']:
            landmarks = frame_data['landmarks']
            
            #map mediapipe landmark indices to joint names
            joint_mapping = {
                'nose': 0,
                'left_shoulder': 11,
                'right_shoulder': 12,
                'left_elbow': 13,
                'right_elbow': 14,
                'left_wrist': 15,
                'right_wrist': 16,
                'left_hip': 23,
                'right_hip': 24,
                'left_knee': 25,
                'right_knee': 26,
                'left_ankle': 27,
                'right_ankle': 28
            }
            
            frame_width = 1.0  #will be updated later
            frame_height = 1.0
            
            for joint_name, idx in joint_mapping.items():
                if idx < len(landmarks):
                    landmark = landmarks[idx]
                    if i == 0 and joint_name == 'left_shoulder': 
                        print(f"Debug landmark data for {joint_name}: {landmark}")
                    
                    #normalized coordinates store
                    if landmark.get('visibility', 0) > 0.5:
                        frame_keypoints[joint_name] = [
                            float(landmark['x']), 
                            float(landmark['y']), 
                            float(landmark.get('visibility', 1.0))
                        ]
        
        keypoints_per_frame.append(frame_keypoints)

if 'angles' in swing_data:
    angles_data = swing_data['angles']
    if isinstance(angles_data, dict):
        #find the length of angle data
        angle_length = 0
        for key, values in angles_data.items():
            if key != 'timestamps' and isinstance(values, list):
                angle_length = max(angle_length, len(values))
        
        print(f"Processing {angle_length} angle frames...")
        
        for i in range(angle_length):
            frame_angles = {}
            for angle_name, values in angles_data.items():
                if angle_name != 'timestamps' and isinstance(values, list) and i < len(values):
                    if not np.isnan(values[i]):  #skip NaN values
                        frame_angles[angle_name] = values[i]
            angles_per_frame.append(frame_angles)

swing_phases = swing_data.get('swing_phases', {})

print(f"Converted data:")
print(f"Keypoints per frame: {len(keypoints_per_frame)}")
print(f"Angles per frame: {len(angles_per_frame)}")
print(f"Swing phases: {swing_phases}")

#bones for drawing lines
bones = [
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    #torso connections
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip')
]
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    print(f"File exists: {video_path.exists()}")
    exit(1)

#get actual video dimensions
actual_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#test reading first frame
ret, test_frame = cap.read()
if not ret:
    print("Error: Could not read first frame")
    exit(1)
else:
    print(f"Video dimensions: {actual_frame_width}x{actual_frame_height}")
    print(f"Actual frame shape: {test_frame.shape}")
    
#reset to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video frames: {frame_count}")
print(f"Keypoint frames: {len(keypoints_per_frame)}")

print("\nStarting visualization. Controls:")
print("- 'q': quit")
print("- 'space': pause/resume") 
print("- 'r': restart from beginning")
print("- Left/Right arrows: step frame by frame when paused")

paused = False
frame_idx = 0
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

delay = int(1000 / fps) if fps > 0 else 33  #calculate delay in ms for real-time playback

print(f"Video FPS: {fps}, using {delay}ms delay between frames")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached - restarting from beginning")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            continue
        
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    
    #ensure we don't go out of bounds
    if frame_idx >= len(keypoints_per_frame):
        print(f"Frame {frame_idx} out of range for keypoint data")
        continue
    
    keypoints = keypoints_per_frame[frame_idx]
    angles = angles_per_frame[frame_idx] if frame_idx < len(angles_per_frame) else {}
    
    #add frame info
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if frame_idx == 0:
        print(f"Frame 0 keypoints: {keypoints}")
        for name, coord in list(keypoints.items())[:2]:
            print(f"  {name}: {coord}")
    
    #draw keypoints w/ coordinate conversion and 90-degree left rotation
    for name, coord in keypoints.items():
        if len(coord) >= 2:
            try:
                norm_x = float(coord[0])
                norm_y = float(coord[1])
                
                #rotate 90 degrees left without flipping upside down
                #mapping: (x, y)->(1 - y, x)
                rotated_x = 1.0 - norm_y
                rotated_y = norm_x
                
                #convert to pixel coords
                x = int(rotated_x * actual_frame_width)
                y = int(rotated_y * actual_frame_height)
                
                #make sure coordinates are within frame bounds
                x = max(0, min(x, actual_frame_width - 1))
                y = max(0, min(y, actual_frame_height - 1))
                
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(frame, name, (x + 8, y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            except (ValueError, TypeError) as e:
                print(f"Error converting coordinates for {name}: {coord}, error: {e}")
    
    #draw bones with proper coordinate conversion and 90-degree left rotation
    for start, end in bones:
        if start in keypoints and end in keypoints:
            if len(keypoints[start]) >= 2 and len(keypoints[end]) >= 2:
                try:
                    #start point rotation
                    norm_x1 = float(keypoints[start][0])
                    norm_y1 = float(keypoints[start][1])
                    rotated_x1 = 1.0 - norm_y1
                    rotated_y1 = norm_x1
                    x1 = int(rotated_x1 * actual_frame_width)
                    y1 = int(rotated_y1 * actual_frame_height)
                    
                    #end point rotation
                    norm_x2 = float(keypoints[end][0])
                    norm_y2 = float(keypoints[end][1])
                    rotated_x2 = 1.0 - norm_y2
                    rotated_y2 = norm_x2
                    x2 = int(rotated_x2 * actual_frame_width)
                    y2 = int(rotated_y2 * actual_frame_height)
                    
                    # Clamp to frame bounds
                    x1 = max(0, min(x1, actual_frame_width - 1))
                    y1 = max(0, min(y1, actual_frame_height - 1))
                    x2 = max(0, min(x2, actual_frame_width - 1))
                    y2 = max(0, min(y2, actual_frame_height - 1))
                    
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                except (ValueError, TypeError) as e:
                    print(f"Error drawing bone {start}-{end}: {e}")
    
    #annotate angles
    y_offset = 60
    for angle_name, angle_val in angles.items():
        cv2.putText(frame, f"{angle_name}: {angle_val:.1f} deg", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
    
    out.write(frame)
    cv2.imshow("Swing Analysis Overlay", frame)
    
    #use proper delay based on video FPS, but allow for quick response
    key = cv2.waitKey(max(1, delay)) & 0xFF
    if key == ord('q'):
        print("User pressed 'q' to quit")
        break
    elif key == ord(' '): #space bar to pause an start
        paused = not paused
        print("Paused" if paused else "Resumed")
    elif key == ord('r'):  #r to restart
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("Restarting video")
    elif paused:
        # Frame stepping when paused
        if key == 83 or key == 2:  #right arrow
            ret, frame = cap.read()
            if ret:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                print(f"Step forward to frame {frame_idx}")
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print("Reached end, restarting")
        elif key == 81 or key == 0:  #left arrow
            current_pos = max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            ret, frame = cap.read()
            if ret:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                print(f"Step back to frame {frame_idx}")
    
    #shhow current progress
    if frame_idx % 10 == 0:  #every 10th frame
        progress = (frame_idx / frame_count) * 100
        print(f"Progress: {progress:.1f}% (Frame {frame_idx}/{frame_count})")

print("Video playback finished or interrupted")
print("Final frame index:", frame_idx)

#keep the last frame visible
print("Press any key in the OpenCV window to close, or 'q' to quit immediately")
while True:
    key = cv2.waitKey(0) & 0xFF  #wait indefinitely for key press
    if key == ord('q') or key == 27:  #q or esc
        break
    print(f"Key pressed: {chr(key) if key < 128 else key}. Press 'q' to quit.")

out.release()
cap.release()
print(f"Video saved at: {output_path}")
cv2.destroyAllWindows()
print("Visualization complete!")