#!/usr/bin/env python3
"""
Test script to demonstrate the SwingCoach functionality with badminton-specific metrics
"""

from analysis import SwingCoach
import json

def test_coach():
    # Initialize the coach (will use OPENAI_API_KEY environment variable)
    coach = SwingCoach()
    
    # Sample badminton analysis data (similar to what the API returns)
    sample_analysis = {
        "filename": "badminton_swing.mov",
        "frame_count": 53,
        "fps": 30.89,
        "swing_phases": {
            "setup": 0,
            "racket_prep": 15,
            "backswing_start": 26,
            "top_of_backswing": 31,
            "downswing_start": 35,
            "contact": 39,
            "follow_through": 52
        },
        "similarity_scores": {
            "right_elbow": 0.85,
            "left_shoulder": 0.72,
            "spine_angle": 0.91,
            "shoulder_rotation": 0.78,
            "wrist_angle": 0.82,
            "racket_prep_angle": 0.75
        },
        "angles": {
            "spine_angle": [90] * 53,  # Simplified for testing
            "shoulder_rotation": [95] * 53,
            "hip_rotation": [88] * 53,
            "wrist_angle": [85] * 53,
            "racket_prep_angle": [92] * 53
        },
        "contact_height": 0.45  # Badminton-specific metric
    }
    
    print("Testing Badminton SwingCoach with sample data...")
    print(f"Sample data: {json.dumps(sample_analysis, indent=2)}")
    
    try:
        # Get swing advice
        advice = coach.get_swing_advice(sample_analysis)
        
        print("\n🏸 Badminton AI Coach Advice:")
        print(f"Summary: {advice.get('summary', 'N/A')}")
        
        if 'priority_issues' in advice:
            print("\nPriority Issues:")
            for issue in advice['priority_issues']:
                print(f"  • {issue}")
        
        if 'specific_tips' in advice:
            print("\nSpecific Tips:")
            for tip in advice['specific_tips']:
                print(f"  • {tip}")
        
        if 'positive_feedback' in advice:
            print("\nPositive Feedback:")
            for feedback in advice['positive_feedback']:
                print(f"  • {feedback}")
        
        if 'drill_suggestions' in advice:
            print("\nPractice Drills:")
            for drill in advice['drill_suggestions']:
                print(f"  • {drill}")
        
        if 'computed_flags' in advice:
            print(f"\nComputed Badminton Flags: {advice['computed_flags']}")
        
        if 'cache_key' in advice:
            print(f"Cache Key: {advice['cache_key']}")
            
    except Exception as e:
        print(f"Error getting advice: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set")

def test_badminton_phases():
    """Test the badminton-specific swing phases"""
    print("\n🏸 Testing Badminton Swing Phases:")
    
    # Test the new badminton phases
    phases = {
        "setup": 0,
        "racket_prep": 15,
        "backswing_start": 26,
        "top_of_backswing": 31,
        "downswing_start": 35,
        "contact": 39,
        "follow_through": 52
    }
    
    print("Badminton Swing Phases:")
    for phase, frame in phases.items():
        print(f"  {phase}: frame {frame}")
    
    # Test timing calculations
    racket_prep_time = phases['racket_prep'] - phases['setup']
    backswing_time = phases['top_of_backswing'] - phases['backswing_start']
    downswing_time = phases['contact'] - phases['downswing_start']
    follow_through_time = phases['follow_through'] - phases['contact']
    
    print(f"\nTiming Analysis:")
    print(f"  Racket Prep: {racket_prep_time} frames")
    print(f"  Backswing: {backswing_time} frames")
    print(f"  Downswing: {downswing_time} frames")
    print(f"  Follow Through: {follow_through_time} frames")

if __name__ == "__main__":
    test_coach()
    test_badminton_phases() 