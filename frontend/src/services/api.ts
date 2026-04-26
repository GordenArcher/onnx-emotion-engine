import { APIResponse } from "../types/envelope";

export async function predictEmotion(
  pixelArray: number[],
): Promise<APIResponse> {
  const API_BASE = import.meta.env.VITE_API_URL;

  const response = await fetch(`${API_BASE}/api/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pixels: pixelArray }),
  });

  const data: APIResponse = await response.json();

  if (data.status === "error") {
    throw new Error(data.message || "Prediction failed");
  }

  return data;
}
