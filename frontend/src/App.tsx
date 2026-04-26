import { useState, useCallback, useRef } from "react";
import Webcam from "./components/Webcam";
import ResultDisplay from "./components/ResultDisplay";
import { predictEmotion } from "./services/api";
import { softmax } from "./utils/math";
import { EmotionData, Metadata } from "./types/envelope";
import { useServerPing } from "./hooks/useServerPing";

function App() {
  const [result, setResult] = useState<EmotionData | null>(null);
  const [requestID, setsetRequestID] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isProcessingRef = useRef<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);

  const handleFaceCaptured = useCallback(async (pixels: number[]) => {
    if (isProcessingRef.current) return;
    isProcessingRef.current = true;
    setIsProcessing(true);

    try {
      const response = await predictEmotion(pixels);

      if (response.data) {
        const rawProbs = Object.values(response.data.probabilities);
        const normalizedProbs = softmax(rawProbs);

        const finalProbs: Record<string, number> = {};
        Object.keys(response.data.probabilities).forEach((key, index) => {
          finalProbs[key] = normalizedProbs[index];
        });

        setsetRequestID(response.request_id);

        setResult({ ...response.data, probabilities: finalProbs });
      }

      setMetadata(response.metadata);
      setError(null);
    } catch (err) {
      if (err instanceof Error) setError(err.message);
    } finally {
      isProcessingRef.current = false;
      setIsProcessing(false);
    }
  }, []);

  // Render's free tier spins down containers after 15 minutes of inactivity.
  // A cold start takes 30-60 seconds—long enough for users to close the tab.
  // This hook pings /api/v1/health every 2 minutes to keep the ONNX runtime
  // warm in memory, ensuring sub-10ms inference even on the free plan. DO NOT REMOVE
  useServerPing();

  return (
    <div className="max-w-7xl mx-auto p-4 md:p-6 lg:p-8 min-h-screen bg-slate-950 text-slate-300 flex flex-col">
      <header className="mb-4 md:mb-5">
        <h1 className="text-xl md:text-2xl font-semibold text-white m-0">
          ONNX Emotion Engine
        </h1>
        <p className="text-xs md:text-sm font-mono text-slate-500 mt-1">
          React → Go (ONNX) → MobileNetV2
        </p>
      </header>

      <main className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-5 min-h-0">
        <section className="bg-slate-900 border border-slate-700 rounded-md overflow-hidden flex flex-col min-h-[300px] md:min-h-0">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 p-3 border-b border-slate-700 bg-slate-950 shrink-0">
            Live Feed (Native Face Detection)
          </div>
          <Webcam
            onFaceCaptured={handleFaceCaptured}
            isProcessing={isProcessing}
          />
        </section>

        <section className="bg-slate-900 border border-slate-700 rounded-md overflow-hidden flex flex-col min-h-[300px] md:min-h-0">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 p-3 border-b border-slate-700 bg-slate-950 shrink-0">
            Inference Results
          </div>

          {error && (
            <div className="m-3 md:m-4 p-2.5 bg-red-400/10 border border-red-400/40 text-red-400 text-[13px] rounded">
              {error}
            </div>
          )}

          <ResultDisplay
            data={result}
            metadata={metadata}
            request_id={requestID}
          />
        </section>
      </main>
    </div>
  );
}

export default App;
