from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
from analysis import SwingAnalyzer, SwingCoach
import json
import numpy as np
from typing import Any, Dict

#custom JSON encoder that handles NumPy types automatically
class NumpyJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return super().render(self.convert_numpy_types(content))
    
    @staticmethod
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: NumpyJSONResponse.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [NumpyJSONResponse.convert_numpy_types(item) for item in obj]
        else:
            return obj
        
app = FastAPI(title="Swingalyze")

analyzer = SwingAnalyzer()
coach = SwingCoach()  #will use OPENAI_API_KEY environment variable

BASE_DIR = Path(__file__).parent.resolve()

#global reference data storage
reference_data = None
reference_video_path = None

#Recursively convert numpy types to native Python types for JSON serialization
def to_python_type(obj):
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

def load_default_reference():
    global reference_data, reference_video_path
    
    reference_path = BASE_DIR / "reference.MOV"
    if reference_path.exists():
        try:
            print(f"Loading default reference video: {reference_path}")
            reference_data = analyzer.analyze_video_detailed(str(reference_path))
            reference_video_path = str(reference_path)
            
            #save reference data for faster loading
            reference_data_path = BASE_DIR / "reference_data.json"
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

#load default reference video on startup
@app.on_event("startup")
async def startup_event():
    load_default_reference()

@app.get("/")
async def root():
    return {"message": "Swingalyze API is running"}

@app.post("/upload-reference")
async def upload_reference_video(file: UploadFile = File(...)):
    global reference_data, reference_video_path
    
    #validate file type
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    #save uploaded file temporarily
    temp_dir = BASE_DIR / "temp_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    temp_file_path = temp_dir / f"reference_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        #analyze reference video
        print(f"Analyzing reference video: {temp_file_path}")
        reference_data = analyzer.analyze_video_detailed(str(temp_file_path))
        reference_video_path = str(temp_file_path)
        
        #save reference data
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
        #clean up on error
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing reference video: {str(e)}")

@app.post("/analyze")
async def analyze_swing(file: UploadFile = File(...)):
    global reference_data, reference_video_path
    
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    temp_dir = BASE_DIR / "temp_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    temp_file_path = temp_dir / f"analysis_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Analyzing video: {temp_file_path}")
        swing_data = analyzer.analyze_video_detailed(str(temp_file_path))
        
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
        
        if reference_data is not None:
            try:
                print("Performing comparison with reference video...")
                comparison = analyzer.compare_swings_robust(
                    [reference_video_path, str(temp_file_path)], 
                    reference_video=reference_video_path
                )
                
                #add comparison data to response
                response_data["comparison"] = {
                    "aligned_swings": comparison['aligned_swings'],
                    "reference_video": reference_video_path
                }
                
                #calculate similarity scores for key angles
                similarity_scores = {}
                if len(comparison['aligned_swings']) >= 2:
                    reference_swing = comparison['aligned_swings'][0]
                    test_swing = comparison['aligned_swings'][1]
                    
                    for angle_name in ['right_elbow', 'left_shoulder', 'spine_angle', 'shoulder_rotation', 'wrist_angle', 'racket_prep_angle']:
                        if angle_name in reference_swing and angle_name in test_swing:
                            ref_angles = np.array(reference_swing[angle_name])
                            test_angles = np.array(test_swing[angle_name])
                            
                            #remove NaN values
                            valid_mask = ~(np.isnan(ref_angles) | np.isnan(test_angles))
                            if np.sum(valid_mask) > 0:
                                ref_valid = ref_angles[valid_mask]
                                test_valid = test_angles[valid_mask]
                                
                                #calculate correlation coefficient as similarity
                                correlation = np.corrcoef(ref_valid, test_valid)[0, 1]
                                similarity_scores[angle_name] = max(0, correlation)  #make sure non-negative
                
                response_data["similarity_scores"] = similarity_scores
                
                #get LLM-powered swing advice
                try:
                    if coach is not None:
                        print("Getting swing advice from AI coach...")
                        advice = coach.get_swing_advice(
                            phases=response_data["swing_phases"],
                            angles=response_data["analysis"]["angles"],
                            similarity_scores=response_data.get("similarity_scores", {}),
                            key_metrics={},
                            reference_metrics={}
                        )
                        response_data["coaching_advice"] = advice
                    else:
                        print("SwingCoach not available, providing basic advice...")
                        response_data["coaching_advice"] = {
                            "summary": "Analysis complete. Review the similarity scores and swing phases.",
                            "priority_issues": ["Check similarity scores below 70%"],
                            "specific_tips": ["Focus on improving the areas with lowest similarity scores"],
                            "positive_feedback": ["Good swing analysis data captured"],
                            "drill_suggestions": ["Practice the swing phases that need improvement"],
                            "note": "AI coaching requires OPENAI_API_KEY environment variable"
                        }
                except Exception as e:
                    print(f"Error getting swing advice: {e}")
                    response_data["coaching_advice"] = {
                        "summary": "Analysis complete. Review the similarity scores and swing phases.",
                        "priority_issues": ["Check similarity scores below 70%"],
                        "specific_tips": ["Focus on improving the areas with lowest similarity scores"],
                        "positive_feedback": ["Good swing analysis data captured"],
                        "drill_suggestions": ["Practice the swing phases that need improvement"],
                        "error": "AI coaching temporarily unavailable"
                    }
            except Exception as comparison_error:
                print(f"Error during comparison: {comparison_error}")
                print(f"Comparison error details: {type(comparison_error).__name__}: {str(comparison_error)}")
                #continue without comparison data
                response_data["comparison_error"] = f"Comparison failed: {str(comparison_error)}"
                response_data["similarity_scores"] = {}
                response_data["coaching_advice"] = {
                    "summary": "Analysis complete but comparison failed. Review the swing phases and angles.",
                    "priority_issues": ["Focus on swing phase timing"],
                    "specific_tips": ["Work on consistent swing phases"],
                    "positive_feedback": ["Good swing analysis data captured"],
                    "drill_suggestions": ["Practice swing phase consistency"],
                    "error": "Comparison analysis unavailable"
                }
        else:
            response_data["message"] = "No reference video available for comparison"
        
        #clean up temporary file
        temp_file_path.unlink()
        
        #convert numpy types to native Python types for JSON serialization
        response_data = to_python_type(response_data)
        
        return response_data
        
    except Exception as e:
        #clean up on error
        if temp_file_path.exists():
            temp_file_path.unlink()
        
        print(f"Error analyzing video: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {type(e).__name__}: {str(e)}")

@app.get("/reference-info")
async def get_reference_info():
    global reference_data, reference_video_path
    
    if reference_data is None:
        return {"message": "No reference video uploaded"}
    
    #convert numpy types to native Python types for JSON serialization
    reference_info = {
        "reference_video": reference_video_path,
        "frame_count": reference_data['frame_count'],
        "fps": reference_data['fps'],
        "swing_phases": reference_data['swing_phases']
    }
    
    return to_python_type(reference_info)

@app.delete("/reference")
async def delete_reference():
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
    success = load_default_reference()
    if success:
        return {"message": "Default reference video reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload default reference video")

#get swing advice for previously analyzed data
@app.post("/get-advice")
async def get_swing_advice(analysis_data: Dict):
    try:
        if coach is not None:
            advice = coach.get_swing_advice(analysis_data)
            return to_python_type(advice)
        else:
            return {
                "error": "SwingCoach not available",
                "message": "Set OPENAI_API_KEY environment variable to enable AI coaching"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting swing advice: {str(e)}")

@app.delete("/coaching-cache")
async def clear_coaching_cache():
    if coach is not None:
        coach.clear_cache()
        return {"message": "Coaching cache cleared"}
    else:
        return {"message": "No coaching cache to clear"}

#cleanup function to remove temp files on shutdown
@app.on_event("shutdown")
async def cleanup():
    temp_dir = BASE_DIR / "temp_uploads"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)