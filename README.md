# ONNX Emotion Engine

A high-performance, real-time facial emotion recognition system built with a decoupled, production-grade architecture. 

Instead of relying on a heavy Python web server at runtime, this engine trains a deep learning model in Python, exports it to ONNX, and executes inference natively in **Go**. The frontend handles face detection via the browser, sending only raw pixel tensors to the backend, achieving sub-10ms inference times.

## Architecture Flow

```text
[ React UI (Webcam) ] 
       │ 1. Captures frame, uses MediaPipe to find & crop face
       │ 2. Converts cropped face (48x48) to a float array
       ▼
[ Go Backend (Gin + ONNX Runtime) ]
       │ 3. Validates payload (strict 6912 float array length)
       │ 4. Runs ONNX inference (~4ms)
       │ 5. Formats standard API response envelope
       ▼
[ React UI ]
       │ 6. Receives dominant emotion & probability distribution
       └──► Updates UI in real-time (30 FPS)
```

## Project Structure

```text
/onnx-emotion-engine
  ├── /ml                 # Python (Offline training only)
  │   ├── train.py        # Model training script
  │   ├── export.py       # PyTorch -> ONNX export script
  │   └── emotion_model.onnx 
  │
  ├── /backend            # Go (Runtime inference engine)
  │   ├── main.go         # Gin HTTP server & standard envelope handler
  │   └── /inference      
  │       └── model.go    # ONNX session initialization & tensor math
  │
  └── /frontend           # React (Client-side UI & processing)
      └── src/
          ├── Webcam.jsx  # MediaPipe face extraction
          └── api.js      # Fetch logic to Go backend
```

## API Standard

All responses follow a strict, enterprise-grade envelope structure:

```json
{
  "status": "success",
  "message": "Emotion inferred successfully",
  "data": {
    "dominant_emotion": "Happy",
    "confidence": 0.94,
    "probabilities": { "Happy": 0.94, "Sad": 0.01, ... }
  },
  "code": "EMOTION_INFERRED",
  "request_id": "uuid-v4",
  "metadata": {
    "inference_time_ms": 4.2,
    "model_version": "v1.0.0",
    "timestamp": "2023-10-27T10:00:00Z"
  }
}
```

## Getting Started

*(Documentation in progress - currently under active development)*

- **Prerequisites:** Python 3.10+, Go 1.21+, Node 18+
- **Step 1:** Train the model in `/ml` and generate the `.onnx` artifact.
- **Step 2:** Start the Go backend server.
- **Step 3:** Launch the React frontend.
