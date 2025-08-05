from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
from analysis import SwingAnalyzer
import json
import numpy as np

app = FastAPI(title="Swingalyze")

# Initialize the swing analyzer
analyzer = SwingAnalyzer()

# Global reference data storage
reference_data = None
reference_video_path = None

def load_default_reference():
    """Load the default reference video if it exists"""
    global reference_data, reference_video_path
    
    reference_path = Path("reference.MOV")
    if reference_path.exists():
        try:
            print(f"Loading default reference video: {reference_path}")
            reference_data = analyzer.analyze_video_detailed(str(reference_path))
            reference_video_path = str(reference_path)
            
            # Save reference data for faster loading
            reference_data_path = Path("reference_data.json")
            if not reference_data_path.exists():
                analyzer.save_reference_data(str(reference_path), str(reference_data_path))
            
            print("Default reference video loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading default reference video: {e}")
            return False
    else:
        print("No default reference video found")
        return False

@app.on_event("startup")
async def startup_event():
    """Load default reference video on startup"""
    load_default_reference()

@app.get("/")
async def root():
    return {"message": "Swingalyze API is running"}

@app.post("/upload-reference")
async def upload_reference_video(file: UploadFile = File(...)):
    """Upload a reference video for comparison"""
    global reference_data, reference_video_path
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    # Save uploaded file temporarily
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    temp_file_path = temp_dir / f"reference_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze reference video
        print(f"Analyzing reference video: {temp_file_path}")
        reference_data = analyzer.analyze_video_detailed(str(temp_file_path))
        reference_video_path = str(temp_file_path)
        
        # Save reference data
        reference_output_path = temp_dir / "reference_data.json"
        analyzer.save_reference_data(str(temp_file_path), str(reference_output_path))
        
        return {
            "message": "Reference video uploaded and analyzed successfully",
            "filename": file.filename,
            "frame_count": reference_data['frame_count'],
            "fps": reference_data['fps'],
            "swing_phases": reference_data['swing_phases']
        }
        
    except Exception as e:
        # Clean up on error
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing reference video: {str(e)}")

@app.post("/analyze")
async def analyze_swing(file: UploadFile = File(...)):
    """Analyze a swing video with optional reference comparison"""
    global reference_data, reference_video_path
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    # Save uploaded file temporarily
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    temp_file_path = temp_dir / f"analysis_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze the uploaded video
        print(f"Analyzing video: {temp_file_path}")
        swing_data = analyzer.analyze_video_detailed(str(temp_file_path))
        
        # Prepare response data
        response_data = {
            "filename": file.filename,
            "frame_count": swing_data['frame_count'],
            "fps": swing_data['fps'],
            "swing_phases": swing_data['swing_phases'],
            "analysis": {
                "angles": swing_data['angles'],
                "normalized_angles": swing_data['normalized_angles']
            }
        }
        
        # If reference data exists, perform comparison
        if reference_data is not None:
            print("Performing comparison with reference video...")
            comparison = analyzer.compare_swings_robust(
                [reference_video_path, str(temp_file_path)], 
                reference_video=reference_video_path
            )
            
            # Add comparison data to response
            response_data["comparison"] = {
                "aligned_swings": comparison['aligned_swings'],
                "reference_video": reference_video_path
            }
            
            # Calculate similarity scores for key angles
            similarity_scores = {}
            if len(comparison['aligned_swings']) >= 2:
                reference_swing = comparison['aligned_swings'][0]
                test_swing = comparison['aligned_swings'][1]
                
                for angle_name in ['right_elbow', 'left_shoulder', 'spine_angle', 'shoulder_rotation']:
                    if angle_name in reference_swing and angle_name in test_swing:
                        ref_angles = np.array(reference_swing[angle_name])
                        test_angles = np.array(test_swing[angle_name])
                        
                        # Remove NaN values
                        valid_mask = ~(np.isnan(ref_angles) | np.isnan(test_angles))
                        if np.sum(valid_mask) > 0:
                            ref_valid = ref_angles[valid_mask]
                            test_valid = test_angles[valid_mask]
                            
                            # Calculate correlation coefficient as similarity
                            correlation = np.corrcoef(ref_valid, test_valid)[0, 1]
                            similarity_scores[angle_name] = max(0, correlation)  # Ensure non-negative
            
            response_data["similarity_scores"] = similarity_scores
        else:
            response_data["message"] = "No reference video available for comparison"
        
        # Clean up temporary file
        temp_file_path.unlink()
        
        return response_data
        
    except Exception as e:
        # Clean up on error
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {str(e)}")

@app.get("/reference-info")
async def get_reference_info():
    """Get information about the current reference video"""
    global reference_data, reference_video_path
    
    if reference_data is None:
        return {"message": "No reference video uploaded"}
    
    return {
        "reference_video": reference_video_path,
        "frame_count": reference_data['frame_count'],
        "fps": reference_data['fps'],
        "swing_phases": reference_data['swing_phases']
    }

@app.delete("/reference")
async def delete_reference():
    """Delete the current reference video and data"""
    global reference_data, reference_video_path
    
    if reference_video_path and os.path.exists(reference_video_path):
        try:
            os.remove(reference_video_path)
        except:
            pass
    
    reference_data = None
    reference_video_path = None
    
    return {"message": "Reference video deleted"}

@app.post("/reload-reference")
async def reload_reference():
    """Reload the default reference video"""
    success = load_default_reference()
    if success:
        return {"message": "Default reference video reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload default reference video")

# Cleanup function to remove temporary files on shutdown
@app.on_event("shutdown")
async def cleanup():
    temp_dir = Path("temp_uploads")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)