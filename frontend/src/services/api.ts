import axios from "axios";
import { APIResponse } from "../types/envelope";

const API_BASE = import.meta.env.PROD
  ? (import.meta.env.VITE_API_URL ?? "")
  : "";

// Axios automatically stringifies arrays into JSON and handles the network layer. We use it instead of fetch
// native `fetch()` to get automatic JSON parsing and proper HTTP error handling that fits our Go envelope.
export async function predictEmotion(
  pixelArray: number[],
): Promise<APIResponse> {
  try {
    const response = await axios.post(`${API_BASE}/api/v1/predict`, {
      pixels: pixelArray,
    });

    // response.data is automatically parsed into our strict TypeScript interfaces.
    const data: APIResponse = // use fallback value in case the API sends garbage.
      response.data?.data
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

    return data;
  } catch (error) {
    // If it's an AxiosError, we dig into the raw error object to pull out our custom Go API response
    // instead of just throwing a generic "Network Error".
    if (axios.isAxiosError(error) && error.response?.data) {
      const data = error.response.data as APIResponse;
      throw new Error(data.message || "Prediction failed", { cause: error });
    }
    throw new Error("Network error or unexpected API failure.", {
      cause: error,
    });
  }
}
