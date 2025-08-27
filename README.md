# Badminton Swing Analyzer

A mobile app that provides AI-powered feedback on badminton swing technique through video analysis.

## What it does

Upload a video of your badminton swing and get personalized AI-powered coaching feedback including technique issues, improvement tips, and practice drills.

## Features

- **Video Upload**: Record or upload badminton swing videos
- **AI Pose Detection**: Extracts 33 body landmarks using MediaPipe
- **Biomechanical Analysis**: Calculates 12+ joint angles and measurements
- **Swing Phase Detection**: Automatically identifies swing phases using ML
- **AI Coaching**: GPT-4o-mini generates personalized feedback
- **Mobile-Friendly**: Designed for smartphone use

## How it works

1. **Record**: Take a video of your badminton swing
2. **Upload**: Send video through the mobile app
3. **Analysis**: AI extracts pose data and calculates biomechanics
4. **Feedback**: Receive coaching advice with issues, tips, and drills

## Tech Stack

- **Frontend**: Expo Go, React Native
- **Backend**: FastAPI with RESTful endpoints
- **AI/ML**: MediaPipe pose detection, custom ML algorithms
- **LLM**: GPT-4o-mini for coaching feedback
- **Computer Vision**: OpenCV, pose estimation

## API Endpoints

- `POST /analyze` - Upload swing video and get AI analysis
- `GET /reference-info` - Get information about current reference video
- `POST /reload-reference` - Reload the reference video data
- `POST /upload-reference` - Upload a new reference video for comparison

## Setup

### Backend

1. Clone the repository
2. Navigate to backend directory:
   ```bash
   cd backend
   ```
3. Create and activate virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate 
   ```
4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. Set environment variables:
   ```bash
   export OPENAI_API_KEY="openai_key"
   ```
6. Run the server:
   ```bash
   python3 main.py
   ```

### Frontend (Mobile)

1. Navigate to frontend directory:
    ```bash
    cd frontend
    ```
2. Install dependencies:
    ```bash
    npm install
    ```
3. Start the Expo development server:
    ```bash
    npx expo start
    ```
4. Use Expo Go app to scan QR code and run on your phone

## Example Response

```json
{
  "summary": "Your swing shows good wrist snap but needs work on racket preparation",
  "priority_issues": ["Racket preparation too rushed", "Insufficient shoulder rotation"],
  "specific_tips": ["Take time to position racket before backswing", "Rotate shoulders 90+ degrees"],
  "positive_feedback": ["Good wrist snap at contact", "Consistent follow through"],
  "drill_suggestions": ["Racket preparation exercises", "Shoulder rotation drills"]
}
```
