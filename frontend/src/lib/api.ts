import type { AnalyzeResponse } from "../types/analysis";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function analyzeTicket(ticketText: string): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ticket_text: ticketText,
      top_k: 5,
    }),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<AnalyzeResponse>;
}
