from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI(title="Swingalyze")

#GET endpoint for root path returning JSON message
#async def help function run asynchronously for better performance
@app.get("/")
async def root():
    return{"message": "Swingalyze API is running"}


#POST endpoint at /analyze that expectes a file upload(the swing video)
@app.post("/analyze")
async def analyze_swing(file: UploadFile = File(...)):
    print("Received request")
    #return mock feedback for now
    return {
        "score": 85,
        "feedback": ["Good swing!"],
        "filename": file.filename
    }

#runs when directly executing this file to start server (used for testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)