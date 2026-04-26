import axios from "axios";
import { APIResponse, BatchAPIResponse } from "../types/envelope";
import { track } from "@vercel/analytics";

const API_BASE = import.meta.env.VITE_API_URL;

// Axios automatically stringifies arrays into JSON and handles the network
// layer. We use it instead of native fetch() to get automatic JSON parsing
// and proper HTTP error handling that fits our Go envelope.
export async function predictEmotion(
  pixelArray: number[],
): Promise<APIResponse> {
  try {
    const response = await axios.post(`${API_BASE}/api/v1/predict`, {
      pixels: pixelArray,
    });

    // response.data is automatically parsed into our strict TypeScript
    // interfaces. We validate the shape defensively — if the API sends
    // garbage (empty body, wrong content-type, upstream proxy error), we
    // fall back to a well-formed error envelope rather than letting the
    // caller deal with undefined field access.
    const data: APIResponse = response.data?.data
      ? response.data
      : {
          status: "error",
          message: "Invalid response structure.",
          data: null,
          errors: null,
          code: "INVALID_RESPONSE",
          request_id: "unknown",
          metadata: null,
        };

    if (data.status === "error") {
      throw new Error(data.message);
    }

    track("prediction_made", {
      dominant_emotion: data?.data.dominant_emotion,
      confidence: parseFloat(data?.data.confidence.toFixed(2)),
      inference_time_ms: data.metadata?.inference_time_ms ?? 0,
    });

    return data;
  } catch (error) {
    // If it's an AxiosError, we dig into the raw error object to pull out
    // our custom Go API response instead of throwing a generic "Network Error".
    if (axios.isAxiosError(error) && error.response?.data) {
      const data = error.response.data as APIResponse;
      throw new Error(data.message || "Prediction failed", { cause: error });
    }
    throw new Error("Network error or unexpected API failure.", {
      cause: error,
    });
  }
}

// predictEmotionBatch sends all detected faces in a single round trip to
// /api/v2/predict/batch. The faces array maps 1:1 with the bounding boxes
// MediaPipe returned — face at index 0 in the request gets result at index 0
// in the response. We rely on the Go server to preserve that ordering, which
// it does because it processes faces sequentially and appends results in the
// same loop order.
//
// We track a batch_prediction_made event rather than N individual
// prediction_made events so the Vercel analytics dashboard doesn't inflate
// prediction counts when multiple faces are visible.
export async function predictEmotionBatch(
  faces: number[][],
): Promise<BatchAPIResponse> {
  try {
    const response = await axios.post(`${API_BASE}/api/v2/predict/batch`, {
      faces,
    });

    const data: BatchAPIResponse = response.data?.data
      ? response.data
      : {
          status: "error",
          message: "Invalid batch response structure.",
          data: null,
          errors: null,
          code: "INVALID_RESPONSE",
          request_id: "unknown",
          metadata: null,
        };

    if (data.status === "error") {
      throw new Error(data.message);
    }

    track("batch_prediction_made", {
      face_count: data.data?.face_count ?? 0,
      inference_time_ms: data.metadata?.inference_time_ms ?? 0,
    });

    return data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.data) {
      const data = error.response.data as BatchAPIResponse;
      throw new Error(data.message || "Batch prediction failed", {
        cause: error,
      });
    }
    throw new Error("Network error or unexpected API failure.", {
      cause: error,
    });
  }
}
