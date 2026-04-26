# React Client

This directory contains the user-facing web interface. It is responsible for capturing the webcam stream, extracting the face, formatting the tensor payload, and rendering the real-time inference results received from the Go backend.

## Directory Structure

```text
/frontend
  ├── src/
  │   ├── components/
  │   │   ├── Webcam.jsx       # Manages browser media permissions, MediaPipe face detection, and canvas rendering
  │   │   └── ResultDisplay.jsx# Parses the standard API envelope and renders confidence bars
  │   ├── services/
  │   │   └── api.js           # HTTP client configured to communicate with the Go server
  │   ├── utils/
  │   │   └── math.js          # Pure JavaScript helper functions (e.g., Softmax normalization)
  │   ├── App.jsx              # Root orchestrator that connects the Webcam to the API to the Results
  │   ├── App.css              # Styling for the main layout
  │   ├── main.jsx             # Application entry point
  │   └── index.css            # Global CSS resets
  ├── vite.config.js           # Vite and CORS configuration
  └── package.json
```

## Architecture & Decisions

### Offloading Computer Vision to the Client
In a traditional setup, the frontend would send a full JPEG frame to the backend, and Python/OpenCV would handle face detection. We deliberately avoid this for two reasons:
1. **Latency:** Sending a 1080p JPEG 30 times a second creates massive network overhead. Sending a tiny array of 6,912 floats takes microseconds.
2. **Backend Constraint:** Our Go backend is optimized for pure tensor math via ONNX. Go's image processing libraries are slow compared to Python. 

To solve this, we use Google's `@mediapipe/face_detection` running locally in the browser via WebAssembly. The browser finds the face, crops it, resizes it to 48x48 using native Canvas APIs, extracts the raw RGBA pixels, and converts them to the float array our Go server expects.

### Handling Raw Logits
The Go server returns raw ONNX logits (which can be negative numbers). Instead of forcing Go to run a Softmax function—which would require adding math libraries to our lean Go binary—we do the normalization here in React. A lightweight JavaScript Softmax function processes the logits into 0.0-1.0 probabilities exactly at the moment they are injected into the UI state. This keeps the Go server focused strictly on inference.

### Network Strategy
We do not use WebSockets for this real-time stream. While WebSockets are great for bi-directional chat, our payload is strictly uni-directional (Client -> Server). Standard HTTP `fetch` calls are easier to debug, stateless, and perfectly capable of handling 30 requests per second on a local network without exhausting browser connection limits.

## Running the Client

1. Ensure you are in the `/frontend` directory.
2. Install dependencies: `npm install`
3. Start the Vite dev server: `npm run dev`

The application will be available at `http://localhost:5173`. By default, Vite proxies missing API routes to the Go backend to avoid CORS issues (configured in `vite.config.js`).
