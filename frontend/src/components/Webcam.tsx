import { useEffect, useRef, useState, useCallback } from "react";
import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";
import { motion, AnimatePresence } from "framer-motion";

interface WebcamProps {
  onFaceCaptured: (pixels: number[]) => void;
  isProcessing: boolean;
}

function CustomSelect({
  devices,
  selectedId,
  onChange,
  disabled,
}: {
  devices: MediaDeviceInfo[];
  selectedId: string;
  onChange: (id: string) => void;
  disabled: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const selectedDevice = devices.find((d) => d.deviceId === selectedId);

  return (
    <div
      className="relative"
      ref={(node) => {
        if (!isOpen && node) return;
        const handleClickOutside = (e: MouseEvent) => {
          if (node && !node.contains(e.target as Node)) setIsOpen(false);
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () =>
          document.removeEventListener("mousedown", handleClickOutside);
      }}
    >
      <button
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className="w-full max-w-75 bg-slate-800 border border-slate-700 text-slate-300 text-sm font-mono rounded px-3 py-1.5 text-left truncate outline-none focus:border-blue-500 disabled:opacity-50 cursor-pointer transition-colors"
      >
        {selectedDevice?.label || "Select camera..."}
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            transition={{ duration: 0.15 }}
            className="absolute top-full left-0 mt-1 w-full max-w-75 bg-slate-800 border border-slate-700 rounded shadow-lg z-150 overflow-hidden"
          >
            <div className="max-h-48 overflow-y-auto py-1">
              {devices.map((device) => (
                <button
                  key={device.deviceId}
                  onClick={() => {
                    onChange(device.deviceId);
                    setIsOpen(false);
                  }}
                  className={`w-full text-left px-3 py-2 text-sm font-mono truncate cursor-pointer transition-colors ${
                    device.deviceId === selectedId
                      ? "bg-slate-700 text-blue-400"
                      : "text-slate-300 hover:bg-slate-700/50"
                  }`}
                >
                  {device.label || `Camera ${device.deviceId.slice(0, 8)}...`}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function Webcam({ onFaceCaptured, isProcessing }: WebcamProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Overlay canvas for drawing the bounding box on top of the video feed.
  // We use a separate canvas rather than the capture canvas so the 48×48
  // crop and the full-resolution overlay never interfere with each other.
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number>(0);
  const detectorRef = useRef<FaceDetector | null>(null);

  // Mirror isProcessing into a ref so the frame loop always reads the live
  // value without needing to be recreated every render cycle.
  const isProcessingRef = useRef(isProcessing);
  useEffect(() => {
    isProcessingRef.current = isProcessing;
  }, [isProcessing]);

  // isPaused drives both the UI toggle and the frame loop guard. Stored as a
  // ref so the loop reads it synchronously without a stale closure.
  const isPausedRef = useRef(false);
  const [isPaused, setIsPaused] = useState(false);

  // Confidence threshold, frames where detection confidence falls below this
  // value are silently skipped. Exposed as a slider in the options panel.
  const confidenceThresholdRef = useRef(0.5);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);

  // FPS tracking, we count how many frames are processed per second and
  // expose the rolling value in a dev overlay.
  const fpsFrameCountRef = useRef(0);
  const fpsLastTickRef = useRef(performance.now());
  const [fps, setFps] = useState(0);

  // Whether the options panel (threshold + FPS) is expanded.
  const [showOptions, setShowOptions] = useState(false);

  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraDevices, setCameraDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [isConnecting, setIsConnecting] = useState<boolean>(true);

  // faceDetected drives the bounding box colour, green when a face is present,
  // absent when not. Tracked as state because the overlay needs to react to it.
  const [faceDetected, setFaceDetected] = useState(false);

  const initializeDetector = useCallback(async () => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm",
    );
    const faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.5,
    });
    detectorRef.current = faceDetector;
  }, []);

  const fetchDevices = useCallback(async () => {
    const allDevices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = allDevices.filter((d) => d.kind === "videoinput");
    setCameraDevices(videoInputs);
    return videoInputs;
  }, []);

  // drawOverlay renders a bounding box and confidence label onto the
  // transparent overlay canvas that sits on top of the video element.
  // It reads the video's displayed dimensions (not its intrinsic 640×480)
  // so the box stays locked to the face regardless of how the video is
  // scaled by CSS.
  const drawOverlay = useCallback(
    (
      bbox: { originX: number; originY: number; width: number; height: number },
      confidence: number,
      videoEl: HTMLVideoElement,
    ) => {
      const overlay = overlayCanvasRef.current;
      if (!overlay) return;
      const ctx = overlay.getContext("2d");
      if (!ctx) return;

      // Keep the overlay canvas pixel-perfect with the displayed video size.
      const { clientWidth, clientHeight } = videoEl;
      if (overlay.width !== clientWidth || overlay.height !== clientHeight) {
        overlay.width = clientWidth;
        overlay.height = clientHeight;
      }

      ctx.clearRect(0, 0, overlay.width, overlay.height);

      // Scale factors from the video's intrinsic resolution to its displayed size.
      const scaleX = clientWidth / (videoEl.videoWidth || 640);
      const scaleY = clientHeight / (videoEl.videoHeight || 480);

      const x = bbox.originX * scaleX;
      const y = bbox.originY * scaleY;
      const w = bbox.width * scaleX;
      const h = bbox.height * scaleY;

      // Corner bracket style, looks cleaner than a full rectangle and avoids
      // obscuring the face with a solid border.
      const bracketLen = Math.min(w, h) * 0.2;
      const color = "#22c55e"; // green-500

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;

      // Top-left
      ctx.beginPath();
      ctx.moveTo(x, y + bracketLen);
      ctx.lineTo(x, y);
      ctx.lineTo(x + bracketLen, y);
      ctx.stroke();

      // Top-right
      ctx.beginPath();
      ctx.moveTo(x + w - bracketLen, y);
      ctx.lineTo(x + w, y);
      ctx.lineTo(x + w, y + bracketLen);
      ctx.stroke();

      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(x, y + h - bracketLen);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x + bracketLen, y + h);
      ctx.stroke();

      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(x + w - bracketLen, y + h);
      ctx.lineTo(x + w, y + h);
      ctx.lineTo(x + w, y + h - bracketLen);
      ctx.stroke();

      // Confidence label above the top-left bracket
      ctx.shadowBlur = 0;
      ctx.font = "11px monospace";
      ctx.fillStyle = color;
      ctx.fillText(`${(confidence * 100).toFixed(0)}%`, x + 2, y - 5);
    },
    [],
  );

  const clearOverlay = useCallback(() => {
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, overlay.width, overlay.height);
  }, []);

  // updateFps is called once per processed frame. It increments the frame
  // counter and recalculates the rolling FPS value every second.
  const updateFps = useCallback(() => {
    fpsFrameCountRef.current += 1;
    const now = performance.now();
    const elapsed = now - fpsLastTickRef.current;
    if (elapsed >= 1000) {
      setFps(Math.round((fpsFrameCountRef.current * 1000) / elapsed));
      fpsFrameCountRef.current = 0;
      fpsLastTickRef.current = now;
    }
  }, []);

  // processFrameRef holds the frame loop function by reference so that
  // requestAnimationFrame always schedules the current version, preventing
  // stale closures from freezing isProcessing / isPaused reads.
  const processFrameRef = useRef<() => void>(null!);

  processFrameRef.current = () => {
    const video = videoRef.current;

    // If paused, stop the loop entirely. It will be restarted by the toggle.
    if (isPausedRef.current) return;

    if (!video || video.readyState < 2 || video.paused || video.ended) {
      animationFrameRef.current = requestAnimationFrame(
        processFrameRef.current,
      );
      return;
    }

    // While an API call is in-flight, keep the loop alive but skip detection
    // to avoid flooding the server with concurrent requests.
    if (isProcessingRef.current) {
      animationFrameRef.current = requestAnimationFrame(
        processFrameRef.current,
      );
      return;
    }

    const detector = detectorRef.current;
    if (detector) {
      try {
        const results = detector.detectForVideo(video, performance.now());

        if (results.detections.length > 0) {
          const detection = results.detections[0];
          const bbox = detection.boundingBox;
          const confidence = detection.categories?.[0]?.score ?? 0;

          // Skip frames that don't meet the user-configured confidence bar.
          if (confidence < confidenceThresholdRef.current || !bbox) {
            setFaceDetected(false);
            clearOverlay();
          } else {
            setFaceDetected(true);
            drawOverlay(bbox, confidence, video);

            const canvas = canvasRef.current;
            if (!canvas) return;
            const ctx = canvas.getContext("2d", { willReadFrequently: true });
            if (!ctx) return;

            canvas.width = 48;
            canvas.height = 48;
            ctx.drawImage(
              video,
              bbox.originX,
              bbox.originY,
              bbox.width,
              bbox.height,
              0,
              0,
              48,
              48,
            );

            const rgba = ctx.getImageData(0, 0, 48, 48).data;
            const pixels = new Float64Array(6912);
            let i = 0;
            for (let p = 0; p < rgba.length; p += 4) {
              pixels[i++] = rgba[p] / 255.0;
              pixels[i++] = rgba[p + 1] / 255.0;
              pixels[i++] = rgba[p + 2] / 255.0;
            }

            updateFps();
            onFaceCaptured(Array.from(pixels));
          }
        } else {
          setFaceDetected(false);
          clearOverlay();
        }
      } catch {
        // Silent drop on individual frame failures, a single bad frame
        // should never kill the loop.
      }
    }

    setTimeout(() => {
      animationFrameRef.current = requestAnimationFrame(
        processFrameRef.current,
      );
    }, 66);
  };

  const startCamera = useCallback(
    async (deviceId?: string) => {
      setIsConnecting(true);
      setCameraError(null);

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      cancelAnimationFrame(animationFrameRef.current);

      try {
        const constraints: MediaStreamConstraints = {
          video: deviceId
            ? { deviceId: { exact: deviceId }, width: 640, height: 480 }
            : { width: 640, height: 480, facingMode: "user" },
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        const devices = await fetchDevices();
        const activeId =
          stream.getVideoTracks()[0].getSettings().deviceId ?? "";
        setSelectedDeviceId(
          devices.find((d) => d.deviceId === activeId)
            ? activeId
            : (devices[0]?.deviceId ?? ""),
        );

        setIsConnecting(false);

        // Reset pause state when a new camera session starts so the user
        // doesn't get stuck in a paused loop after reconnecting.
        isPausedRef.current = false;
        setIsPaused(false);

        animationFrameRef.current = requestAnimationFrame(
          processFrameRef.current,
        );
      } catch (err) {
        console.error("Camera error:", err);
        setIsConnecting(false);
        setCameraError(
          "Camera access denied or no camera detected. Please check permissions.",
        );
      }
    },
    [fetchDevices],
  );

  const handleDeviceChange = (newDeviceId: string) => {
    cancelAnimationFrame(animationFrameRef.current);
    startCamera(newDeviceId);
  };

  const handleReconnect = () => {
    cancelAnimationFrame(animationFrameRef.current);
    startCamera(selectedDeviceId || undefined);
  };

  const handleTogglePause = () => {
    const next = !isPausedRef.current;
    isPausedRef.current = next;
    setIsPaused(next);

    if (!next) {
      // Resuming, restart the frame loop.
      clearOverlay();
      animationFrameRef.current = requestAnimationFrame(
        processFrameRef.current,
      );
    } else {
      // Pausing, cancel any pending frame and clear the overlay so the
      // bounding box doesn't freeze in place.
      cancelAnimationFrame(animationFrameRef.current);
      clearOverlay();
      setFaceDetected(false);
    }
  };

  const handleConfidenceChange = (value: number) => {
    confidenceThresholdRef.current = value;
    setConfidenceThreshold(value);
  };

  useEffect(() => {
    const boot = async () => {
      try {
        await initializeDetector();
        await startCamera();
      } catch {
        setIsConnecting(false);
        setCameraError("Failed to load face detection model.");
      }
    };
    boot();

    return () => {
      cancelAnimationFrame(animationFrameRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex flex-col h-full">
      <div className="relative z-10 flex items-center justify-between p-3 border-b border-slate-700 bg-slate-950 gap-3">
        <CustomSelect
          devices={cameraDevices}
          selectedId={selectedDeviceId}
          onChange={handleDeviceChange}
          disabled={isConnecting}
        />

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowOptions((v) => !v)}
            disabled={isConnecting}
            title="Detection options"
            className={`px-3 py-1.5 text-sm border rounded transition-colors disabled:opacity-50 cursor-pointer ${
              showOptions
                ? "border-blue-500 text-blue-400 bg-blue-500/10"
                : "border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-500"
            }`}
          >
            Options
          </button>

          <button
            onClick={handleTogglePause}
            disabled={isConnecting}
            title={isPaused ? "Resume detection" : "Pause detection"}
            className={`px-3 py-1.5 text-sm border rounded transition-colors disabled:opacity-50 cursor-pointer ${
              isPaused
                ? "border-amber-500 text-amber-400 bg-amber-500/10 hover:bg-amber-500/20"
                : "border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-500"
            }`}
          >
            {isPaused ? "Resume" : "Pause"}
          </button>

          <button
            onClick={handleReconnect}
            disabled={isConnecting}
            className="px-3 py-1.5 text-sm border border-slate-700 rounded text-slate-400 hover:text-slate-200 hover:border-slate-500 transition-colors disabled:opacity-50 whitespace-nowrap cursor-pointer"
          >
            {isConnecting ? "Connecting..." : "Reconnect"}
          </button>
        </div>
      </div>

      <AnimatePresence>
        {showOptions && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-b border-slate-700 bg-slate-900"
          >
            <div className="flex items-center gap-6 px-4 py-3">
              <div className="flex items-center gap-3 flex-1 min-w-0">
                <label className="text-xs font-mono text-slate-400 whitespace-nowrap">
                  Min confidence
                </label>
                <input
                  type="range"
                  min={0.1}
                  max={0.95}
                  step={0.05}
                  value={confidenceThreshold}
                  onChange={(e) =>
                    handleConfidenceChange(parseFloat(e.target.value))
                  }
                  className="flex-1 accent-blue-500 cursor-pointer"
                />
                <span className="text-xs font-mono text-blue-400 w-8 text-right tabular-nums">
                  {Math.round(confidenceThreshold * 100)}%
                </span>
              </div>

              <div className="flex items-center gap-2 shrink-0">
                <span className="text-xs font-mono text-slate-500">FPS</span>
                <span
                  className={`text-xs font-mono tabular-nums w-6 text-right ${
                    fps >= 12
                      ? "text-green-400"
                      : fps >= 6
                        ? "text-amber-400"
                        : "text-red-400"
                  }`}
                >
                  {isPaused ? "—" : fps}
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="bg-black grow flex items-center justify-center min-h-75 relative overflow-hidden">
        <AnimatePresence mode="wait">
          {cameraError ? (
            <motion.div
              key="camera-error"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="text-center p-5"
            >
              <p className="text-slate-400 font-mono text-sm m-0">
                {cameraError}
              </p>
            </motion.div>
          ) : isConnecting ? (
            <motion.div
              key="connecting"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex items-center justify-center bg-black z-10"
            >
              <p className="text-slate-500 font-mono text-sm animate-pulse">
                Loading AI model & Initializing camera...
              </p>
            </motion.div>
          ) : null}
        </AnimatePresence>

        <video
          ref={videoRef}
          width="640"
          height="480"
          muted
          playsInline
          className={`block w-full ${cameraError ? "hidden" : ""}`}
        />

        {/* Bounding box overlay, sits directly on top of the video and is
            pointer-events:none so it never blocks clicks on the video itself. */}
        <canvas
          ref={overlayCanvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ display: cameraError ? "none" : "block" }}
        />

        <AnimatePresence>
          {isPaused && !cameraError && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="absolute inset-0 flex items-center justify-center bg-black/50 pointer-events-none"
            >
              <span className="font-mono text-amber-400 text-sm border border-amber-500/50 bg-amber-500/10 px-4 py-2 rounded">
                PAUSED
              </span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Face detected indicator, a subtle dot in the corner so the user
            knows detection is live without the bounding box being distracting. */}
        {!cameraError && !isConnecting && !isPaused && (
          <div className="absolute bottom-2 right-2 flex items-center gap-1.5 pointer-events-none">
            <div
              className={`w-2 h-2 rounded-full transition-colors duration-300 ${
                faceDetected ? "bg-green-400" : "bg-slate-600"
              }`}
            />
            <span className="text-xs font-mono text-slate-500">
              {faceDetected ? "face detected" : "no face"}
            </span>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
