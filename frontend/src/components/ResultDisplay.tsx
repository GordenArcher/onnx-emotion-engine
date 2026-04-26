import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BatchFaceResult, Metadata } from "../types/envelope";

interface ResultDisplayProps {
  results: BatchFaceResult[] | null;
  metadata: Metadata | null;
  request_id: string | null;
}

const EMOTION_META: Record<
  string,
  { emoji: string; color: string; dimColor: string }
> = {
  Angry: { emoji: "😠", color: "bg-red-500", dimColor: "bg-red-500/20" },
  Disgust: { emoji: "🤢", color: "bg-lime-500", dimColor: "bg-lime-500/20" },
  Fear: { emoji: "😨", color: "bg-purple-500", dimColor: "bg-purple-500/20" },
  Happy: { emoji: "😄", color: "bg-yellow-400", dimColor: "bg-yellow-400/20" },
  Neutral: { emoji: "😐", color: "bg-slate-400", dimColor: "bg-slate-400/20" },
  Sad: { emoji: "😢", color: "bg-blue-400", dimColor: "bg-blue-400/20" },
  Surprise: {
    emoji: "😲",
    color: "bg-orange-400",
    dimColor: "bg-orange-400/20",
  },
};

const fallback = {
  emoji: "🤔",
  color: "bg-blue-400",
  dimColor: "bg-blue-400/20",
};

// The face tab colors mirror the overlay bracket colors in Webcam.tsx
// Face 1 is green, Face 2 is blue, and so on. Keeping them in sync means
// the user can visually match a result panel to a bounding box on screen
// without having to read the label.
const FACE_COLORS = [
  "border-green-500 text-green-400",
  "border-blue-500 text-blue-400",
  "border-amber-500 text-amber-400",
  "border-pink-500 text-pink-400",
  "border-purple-500 text-purple-400",
];

const FACE_ACTIVE_BG = [
  "bg-green-500/10",
  "bg-blue-500/10",
  "bg-amber-500/10",
  "bg-pink-500/10",
  "bg-purple-500/10",
];

function FaceResult({ result }: { result: BatchFaceResult }) {
  const dominant = result.dominant_emotion;
  const meta = EMOTION_META[dominant] ?? fallback;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-slate-700">
        <h2 className="m-0 text-xl md:text-2xl font-bold text-white flex items-center gap-2">
          <span>{meta.emoji}</span>
          <span>{dominant}</span>
        </h2>
        <span className="font-mono text-xs text-green-400 bg-green-400/10 px-2 py-1 rounded border border-green-400/20">
          {(result.confidence * 100).toFixed(1)}%
        </span>
      </div>

      <div className="p-4 grow">
        <div className="space-y-2.5">
          {Object.entries(result.probabilities).map(
            ([emotion, prob], index) => {
              const isDominant = emotion === dominant;
              const em = EMOTION_META[emotion] ?? fallback;

              return (
                <motion.div
                  key={emotion}
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.25, delay: index * 0.04 }}
                  className={`grid grid-cols-[70px_1fr_55px] md:grid-cols-[80px_1fr_60px] items-center gap-2 md:gap-3 text-[12px] md:text-[13px] rounded px-1 py-0.5 transition-colors ${
                    isDominant ? "bg-white/3" : ""
                  }`}
                >
                  <span
                    className={`text-right font-mono truncate ${
                      isDominant ? "text-white font-semibold" : "text-slate-500"
                    }`}
                  >
                    {emotion}
                  </span>

                  <div
                    className={`h-1.5 rounded-full overflow-hidden ${
                      isDominant ? em.dimColor : "bg-slate-950"
                    }`}
                  >
                    <motion.div
                      className={`h-full rounded-full ${em.color}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.max(prob * 100, 0)}%` }}
                      transition={{
                        duration: 0.35,
                        delay: index * 0.04,
                        ease: "easeOut",
                      }}
                    />
                  </div>

                  <span
                    className={`font-mono text-right ${
                      isDominant ? "text-white font-semibold" : "text-slate-300"
                    }`}
                  >
                    {(prob * 100).toFixed(1)}%
                  </span>
                </motion.div>
              );
            },
          )}
        </div>
      </div>
    </div>
  );
}

export default function ResultDisplay({
  results,
  metadata,
  request_id,
}: ResultDisplayProps) {
  // Active tab index, when multiple faces are detected, the user can switch
  // between result panels. Defaults to 0 (first face). We reset to 0 whenever
  // the face count changes so we never show a stale tab for a face that no
  // longer exists.
  const [activeIndex, setActiveIndex] = useState(0);
  const safeIndex = results ? Math.min(activeIndex, results.length - 1) : 0;

  if (!results || results.length === 0) {
    return (
      <div className="p-10 text-center text-slate-500 font-mono text-sm flex-1 flex items-center justify-center">
        Waiting for prediction...
      </div>
    );
  }

  const activeResult = results[safeIndex];

  return (
    <div className="flex flex-col grow">
      {/* Face selector tabs, only rendered when more than one face is detected.
          For a single face, the tab row would be pointless chrome. */}
      {results.length > 1 && (
        <div className="flex gap-1 px-3 pt-3 border-b border-slate-700">
          {results.map((_, i) => (
            <button
              key={i}
              onClick={() => setActiveIndex(i)}
              className={`px-3 py-1.5 text-xs font-mono rounded-t border-b-2 transition-colors cursor-pointer ${
                i === safeIndex
                  ? `${FACE_COLORS[i % FACE_COLORS.length]} ${FACE_ACTIVE_BG[i % FACE_ACTIVE_BG.length]}`
                  : "border-transparent text-slate-500 hover:text-slate-300"
              }`}
            >
              Face {i + 1}
            </button>
          ))}
        </div>
      )}

      {/* Result panel, AnimatePresence ensures the bars re-animate when
          switching between faces rather than snapping to the new values. */}
      <AnimatePresence mode="wait">
        <motion.div
          key={safeIndex}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="grow"
        >
          <FaceResult result={activeResult} />
        </motion.div>
      </AnimatePresence>

      {metadata && (
        <div className="mt-auto p-4 border-t border-slate-700 font-mono text-[10px] md:text-[11px] text-slate-500 leading-relaxed">
          <div>
            <span className="text-slate-300 inline-block w-16">req_id:</span>
            {request_id}
          </div>
          <div>
            <span className="text-slate-300 inline-block w-16">latency:</span>
            {metadata.inference_time_ms.toFixed(2)} ms
            {results.length > 1 && (
              <span className="text-slate-600 ml-1">
                ({results.length} faces ·{" "}
                {(metadata.inference_time_ms / results.length).toFixed(2)}{" "}
                ms/face)
              </span>
            )}
          </div>
          <div>
            <span className="text-slate-300 inline-block w-16">model:</span>
            {metadata.model_version}
          </div>
        </div>
      )}
    </div>
  );
}
