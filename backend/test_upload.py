#!/usr/bin/env python3
"""
Test script to simulate a video upload to the analyze endpoint
"""

import requests
import os
from pathlib import Path

def test_upload():
    # Test the analyze endpoint
    url = "http://127.0.0.1:8000/analyze"
    
    # Check if reference.MOV exists
    reference_path = Path("reference.MOV")
    if not reference_path.exists():
        print("Error: reference.MOV not found")
        return
    
    # Create a simple test file (or use reference.MOV for testing)
    test_file = reference_path
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('test.mov', f, 'video/quicktime')}
            response = requests.post(url, files=files)
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                print("Upload successful!")
            else:
                print("Upload failed!")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_upload() 