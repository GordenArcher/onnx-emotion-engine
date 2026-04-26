import { APIResponse } from "../types/envelope";

export async function predictEmotion(
  pixelArray: number[],
): Promise<APIResponse> {
  const response = await fetch("/api/v1/predict", {
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
