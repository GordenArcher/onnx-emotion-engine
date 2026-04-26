# ONNX Emotion Engine

A high-performance, real-time facial emotion recognition system built with a decoupled, production-grade architecture.

Instead of relying on a heavy Python web server at runtime, this engine trains a deep learning model in Python, exports it to ONNX, and executes inference natively in **Go**. The frontend handles face detection entirely in the browser via MediaPipe, sending only raw pixel tensors to the backend and achieving sub-10ms inference times on commodity CPU.

**Live demo:** https://emotion-engine-ui.vercel.app

## Architecture Flow

```text
[ React UI (Webcam) ]
       │ 1. Captures frame, uses MediaPipe (blaze_face_short_range) to detect & crop face
       │ 2. Converts cropped 48×48 face to a normalised float32 array (6,912 values)
       ▼
[ Go Backend (Gin + ONNX Runtime) ]
       │ 3. Validates payload — rejects anything that isn't exactly 6,912 floats
       │ 4. Copies pixels into pre-allocated C tensor buffer
       │ 5. Runs ONNX graph (MobileNetV2) — ~4ms on CPU
       │ 6. Wraps result in standard API response envelope
       ▼
[ React UI ]
       │ 7. Receives dominant emotion + full probability distribution
       └──► Updates UI in real-time
```

## Project Structure

```text
/onnx-emotion-engine
  ├── /ml                         # Python — offline training only, never runs in production
  │   ├── train.py                # MobileNetV2 transfer learning on FER-2013
  │   ├── export.py               # PyTorch → ONNX export
  │   └── emotion_model.onnx      # Serialized model artifact
  │
  ├── /backend                    # Go — runtime inference engine
  │   ├── cmd/server/main.go      # Process entrypoint, Gin router, graceful shutdown
  │   ├── internal/
  │   │   ├── handlers/
  │   │   │   └── predict.go      # Request validation, float64→float32, response envelope
  │   │   ├── ml/
  │   │   │   └── inference.go    # ONNX Runtime session lifecycle, pre-allocated C tensors
  │   │   ├── middleware/
  │   │   │   ├── request_id.go   # UUID v4 injection for distributed tracing
  │   │   │   └── cors.go         # Configurable CORS via ALLOWED_ORIGINS env var
  │   │   └── models/
  │   │       └── response.go     # Canonical API envelope (success + error paths)
  │   ├── Dockerfile              # 3-stage build: ORT downloader, Go compiler, slim runtime
  │   ├── docker-compose.yml      # Local development
  │   └── emotion_model.onnx      # Model artifact (copied from /ml)
  │
  └── /frontend                   # React + TypeScript — client-side UI
      └── src/
          ├── components/
          │   ├── Webcam.tsx       # MediaPipe face detection, bounding box overlay, FPS counter
          │   └── ResultDisplay.tsx # Probability bars, dominant emotion, inference metadata
          ├── hooks/
          │   └── useServerPing.ts # Keep-alive ping every 2 minutes (Render free tier)
          └── types/
              └── envelope.ts      # TypeScript interfaces for the Go API envelope
```

## API

All responses follow a strict envelope structure:

```json
{
  "status": "success",
  "message": "Emotion inferred successfully",
  "code": "EMOTION_INFERRED",
  "request_id": "uuid-v4",
  "data": {
    "dominant_emotion": "Happy",
    "confidence": 0.94,
    "probabilities": {
      "Angry": 0.01,
      "Disgust": 0.00,
      "Fear": 0.01,
      "Happy": 0.94,
      "Neutral": 0.02,
      "Sad": 0.01,
      "Surprise": 0.01
    }
  },
  "metadata": {
    "inference_time_ms": 4.2,
    "model_version": "mobilenet-v2-v1.0.0",
    "timestamp": "2026-04-26T10:00:00Z"
  }
}
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Self-documenting API index |
| `GET` | `/api/v1/health` | Health check for load balancers and orchestrators |
| `POST` | `/api/v1/predict` | Run emotion inference on a 6,912-float pixel array |

## Getting Started

### Prerequisites

- Python 3.10+ (model training only)
- Go 1.21+
- Node 18+
- ONNX Runtime 1.21.0 shared library (see `backend/README.md` for platform-specific setup)

### 1. Train and export the model

```bash
cd ml
pip install -r requirements.txt
python train.py
python export.py
cp emotion_model.onnx ../backend/
```

### 2. Start the backend

```bash
cd backend
# Download the ORT shared library for your platform — see backend/README.md
go mod tidy
go run cmd/server/main.go
```

Or with Docker:

```bash
cd backend
docker compose up --build
```

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend proxies `/api` to `http://localhost:8080` in development — no CORS configuration needed locally.

## Deployment

The project deploys to **Render** (backend) and **Vercel** (frontend) via the `render.yaml` blueprint at the repo root.

```bash
# Backend — Render (Docker, free tier)
# Frontend — Vercel (static, auto-deploy on push)
git push origin main
```

See `render.yaml` for the full Render service configuration.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model training | Python, PyTorch, MobileNetV2 |
| Model format | ONNX |
| Backend | Go, Gin, ONNX Runtime 1.21.0 (`yalue/onnxruntime_go` v1.13.0) |
| Frontend | React, TypeScript, Vite, Tailwind CSS, Framer Motion |
| Face detection | MediaPipe Tasks Vision (`blaze_face_short_range`) |
| Containerisation | Docker (3-stage multi-platform build) |
| Backend hosting | Render |
| Frontend hosting | Vercel |
