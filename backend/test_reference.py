#!/usr/bin/env python3
"""
Test script to analyze the reference.MOV file and extract detailed frame-by-frame data
"""

from analysis import SwingAnalyzer
import json
import numpy as np
from pathlib import Path

def to_python_type(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return to_python_type(obj.tolist())
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

def main():
    print("Testing reference video analysis...")
    
    # Initialize analyzer
    analyzer = SwingAnalyzer()
    
    # Check if reference video exists
    reference_path = Path("reference.MOV")
    if not reference_path.exists():
        print("Error: reference.MOV not found in current directory")
        return
    
    try:
        # Analyze the reference video
        print(f"Analyzing {reference_path}...")
        analysis = analyzer.analyze_video_detailed(str(reference_path))
        
        # Print basic info
        print(f"Frame count: {analysis['frame_count']}")
        print(f"FPS: {analysis['fps']:.2f}")
        print(f"Duration: {analysis['frame_count'] / analysis['fps']:.2f} seconds")
        print(f"Swing phases: {analysis['swing_phases']}")
        
        # Save detailed analysis
        output_file = "reference_analysis_detailed.json"
        print(f"Saving detailed analysis to {output_file}...")
        
        serializable_analysis = to_python_type(analysis)
        with open(output_file, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)
        
        print(f"Analysis saved successfully!")
        
        # Print some sample frame data
        print("\nSample frame data:")
        keypoints = analysis['keypoints']
        if keypoints:
            sample_frame = keypoints[0]
            print(f"Frame 0: {sample_frame['frame']}, Timestamp: {sample_frame['timestamp']:.2f}s")
            if sample_frame['landmarks']:
                print(f"Landmarks detected: {len(sample_frame['landmarks'])}")
                print(f"Confidence: {sample_frame['confidence']:.3f}")
                
                # Show some key landmark positions
                landmarks = sample_frame['landmarks']
                key_landmarks = {
                    'nose': landmarks[0],
                    'left_shoulder': landmarks[11],
                    'right_shoulder': landmarks[12],
                    'left_elbow': landmarks[13],
                    'right_elbow': landmarks[14],
                    'left_wrist': landmarks[15],
                    'right_wrist': landmarks[16]
                }
                
                print("\nKey landmark positions (x, y, z, visibility):")
                for name, landmark in key_landmarks.items():
                    print(f"  {name}: ({landmark['x']:.3f}, {landmark['y']:.3f}, {landmark['z']:.3f}, {landmark['visibility']:.3f})")
        
        # Print angle statistics
        print("\nAngle statistics:")
        angles = analysis['angles']
        for angle_name, angle_values in angles.items():
            if angle_name != 'timestamps':
                valid_values = [v for v in angle_values if not np.isnan(v)]
                if valid_values:
                    print(f"  {angle_name}: min={min(valid_values):.1f}°, max={max(valid_values):.1f}°, mean={np.mean(valid_values):.1f}°")
        
        print(f"\nDetailed analysis complete! Check {output_file} for full data.")
        
    except Exception as e:
        print(f"Error analyzing reference video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 