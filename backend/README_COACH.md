# SwingCoach - AI-Powered Badminton Swing Analysis

The SwingCoach class provides personalized badminton swing advice using OpenAI's language models, analyzing swing data to give actionable feedback specific to badminton technique.

## Features

- **Automatic Issue Detection**: Analyzes badminton-specific swing mechanics
- **Personalized Advice**: Uses LLM to generate specific, actionable badminton tips
- **Structured Output**: Returns organized advice in categories (issues, tips, feedback, drills)
- **Badminton-Specific Metrics**: Wrist snap, contact height, racket preparation analysis
- **Caching**: Caches advice to avoid repeated LLM calls for identical data
- **Fallback Support**: Provides standard advice if LLM is unavailable

## Badminton-Specific Features

### Swing Phases
- **Setup**: Initial stance and preparation
- **Racket Prep**: Racket positioning before backswing
- **Backswing Start**: Beginning of backswing motion
- **Top of Backswing**: Maximum backswing position
- **Downswing Start**: Beginning of forward motion
- **Contact**: Shuttle contact point
- **Follow Through**: Completion of swing

### Key Metrics
- **Wrist Snap**: Wrist angle at contact for power generation
- **Contact Height**: Racket height at shuttle contact
- **Racket Prep Angle**: Racket arm positioning before swing
- **Shoulder Rotation**: Upper body rotation for power
- **Hip Rotation**: Lower body positioning and stability

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

## Usage

### Basic Usage

```python
from analysis import SwingCoach

# Initialize and get advice
coach = SwingCoach()
advice = coach.get_swing_advice(analysis_data)
```

### API Integration

The SwingCoach is automatically integrated into the `/analyze` endpoint. When you upload a video:

1. Video is analyzed for keypoints and badminton-specific angles
2. Compared to reference video for similarity scores
3. SwingCoach analyzes the data and provides personalized badminton advice
4. Response includes both technical data and coaching advice

### Response Format

```json
{
  "summary": "Brief swing overview",
  "priority_issues": ["Most important problems to fix"],
  "specific_tips": ["Actionable improvement tips"],
  "positive_feedback": ["Things done well"],
  "drill_suggestions": ["Practice exercises"],
  "computed_flags": {"badminton_swing_flags": "computed_metrics"},
  "cache_key": "unique_hash_for_caching"
}
```

## Badminton Swing Flags

The coach automatically detects issues based on:

- **Timing Issues**: Racket prep, backswing/downswing too fast or slow
- **Angle Problems**: Poor wrist pronation, shoulder rotation, or hip position
- **Similarity Issues**: Low correlation with reference swing
- **Phase Problems**: Unusual badminton swing phase timing
- **Contact Issues**: Poor contact height or racket preparation

## Testing

```bash
python test_coach.py
```

## API Endpoints

- `POST /analyze` - Analyzes video and includes badminton coaching advice
- `POST /get-advice` - Get advice for previously analyzed data
- `DELETE /coaching-cache` - Clear the advice cache

## Mobile App Integration

The mobile app automatically displays:
- 🏸 Badminton AI Coach Advice section with colored-coded tips
- Priority issues (red)
- Specific badminton tips (green)
- Positive feedback (blue)
- Badminton practice drills (orange)

