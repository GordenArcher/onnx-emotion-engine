import { motion } from "framer-motion";
import { EmotionData, Metadata } from "../types/envelope";

interface ResultDisplayProps {
  data: EmotionData | null;
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

export default function ResultDisplay({
  data,
  metadata,
  request_id,
}: ResultDisplayProps) {
  if (!data) {
    return (
      <div className="p-10 text-center text-slate-500 font-mono text-sm flex-1 flex items-center justify-center">
        Waiting for prediction...
      </div>
    );
  }

  const dominant = data.dominant_emotion;
  const meta = EMOTION_META[dominant] ?? fallback;

  return (
    <div className="flex flex-col grow">
      <div className="flex justify-between items-center p-4 md:p-5 border-b border-slate-700">
        <h2 className="m-0 text-2xl md:text-3xl font-bold text-white flex items-center gap-2">
          <span>{meta.emoji}</span>
          <span>{dominant}</span>
        </h2>
        <span className="font-mono text-xs md:text-sm text-green-400 bg-green-400/10 px-2 py-1 rounded border border-green-400/20">
          {(data.confidence * 100).toFixed(1)}%
        </span>
      </div>

      <div className="p-4 grow">
        <div className="space-y-2.5">
          {Object.entries(data.probabilities).map(([emotion, prob], index) => {
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
                  className={`h-1.5 rounded-full overflow-hidden ${isDominant ? em.dimColor : "bg-slate-950"}`}
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
          })}
        </div>
      </div>

      {metadata && (
        <div className="mt-auto p-4 border-t border-slate-700 font-mono text-[10px] md:text-[11px] text-slate-500 leading-relaxed">
          <div>
            <span className="text-slate-300 inline-block w-16">req_id:</span>
            {request_id}
          </div>
          <div>
            <span className="text-slate-300 inline-block w-16">latency:</span>
            {metadata.inference_time_ms.toFixed(2)} ms
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
