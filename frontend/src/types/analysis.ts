export type RetrievedTicket = {
  ticket_id: string;
  brand: string;
  text: string;
  score: number;
  metadata: Record<string, string | number | boolean | null>;
};

export type GeneratedAnswer = {
  answer: string;
  confidence_note: string | null;
  latency_ms: number;
  cost_usd: number;
};

export type MlPrediction = {
  label: string;
  confidence: number;
  model_name: string;
  latency_ms: number;
  cost_usd: number;
};

export type LlmPrediction = {
  label: string;
  confidence: number;
  reason: string;
  latency_ms: number;
  cost_usd: number;
};

export type AnalyzeResponse = {
  request_id: string;
  ticket_text: string;
  retrieval_results: RetrievedTicket[];
  rag_answer: GeneratedAnswer;
  non_rag_answer: GeneratedAnswer;
  ml_prediction: MlPrediction;
  llm_prediction: LlmPrediction;
  recommendation: string;
};
