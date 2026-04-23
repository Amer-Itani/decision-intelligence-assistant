import { AlertCircle, Gauge, Loader2, Search, SendHorizontal } from "lucide-react";
import { useState } from "react";

import { analyzeTicket } from "./lib/api";
import type { AnalyzeResponse, GeneratedAnswer } from "./types/analysis";

const exampleTicket =
  "My internet is down and support keeps saying 24-72 hours. I work from home and need help immediately!!!";

function formatCost(cost: number): string {
  if (cost === 0) {
    return "$0.000000";
  }
  return `$${cost.toFixed(6)}`;
}

function AnswerPanel({
  title,
  answer,
}: {
  title: string;
  answer: GeneratedAnswer;
}) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>{title}</h2>
        <span>{answer.latency_ms.toFixed(2)} ms</span>
      </div>
      <p>{answer.answer}</p>
      <div className="metric-row">
        <span>{answer.confidence_note}</span>
        <strong>{formatCost(answer.cost_usd)}</strong>
      </div>
    </section>
  );
}

function App() {
  const [ticketText, setTicketText] = useState(exampleTicket);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    const trimmedText = ticketText.trim();
    if (trimmedText.length < 3) {
      setError("Enter a support ticket with at least 3 characters.");
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      setResult(await analyzeTicket(trimmedText));
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Request failed.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="top-bar">
          <div>
            <p className="eyebrow">RAG, ML baseline, and LLM comparison</p>
            <h1>Decision Intelligence Assistant</h1>
          </div>
          <div className="status-pill">
            <Gauge size={18} />
            <span>Cost and latency visible</span>
          </div>
        </header>

        <section className="input-band">
          <label htmlFor="ticket">Support ticket</label>
          <textarea
            id="ticket"
            value={ticketText}
            onChange={(event) => setTicketText(event.target.value)}
            rows={5}
          />
          <div className="actions">
            <button type="button" onClick={handleSubmit} disabled={isLoading}>
              {isLoading ? <Loader2 className="spin" size={18} /> : <SendHorizontal size={18} />}
              <span>{isLoading ? "Analyzing" : "Analyze ticket"}</span>
            </button>
          </div>
          {error && (
            <div className="error-row">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}
        </section>

        {result && (
          <>
            <section className="answer-grid">
              <AnswerPanel title="RAG answer" answer={result.rag_answer} />
              <AnswerPanel title="Non-RAG answer" answer={result.non_rag_answer} />
            </section>

            <section className="comparison-band">
              <div className="comparison-card">
                <h2>ML priority</h2>
                <strong>{result.ml_prediction.label}</strong>
                <span>{(result.ml_prediction.confidence * 100).toFixed(0)}% confidence</span>
                <small>
                  {result.ml_prediction.model_name} ·{" "}
                  {result.ml_prediction.latency_ms.toFixed(2)} ms ·{" "}
                  {formatCost(result.ml_prediction.cost_usd)}
                </small>
              </div>
              <div className="comparison-card">
                <h2>LLM zero-shot</h2>
                <strong>{result.llm_prediction.label}</strong>
                <span>{(result.llm_prediction.confidence * 100).toFixed(0)}% confidence</span>
                <small>
                  {result.llm_prediction.latency_ms.toFixed(2)} ms ·{" "}
                  {formatCost(result.llm_prediction.cost_usd)}
                </small>
                <p>{result.llm_prediction.reason}</p>
              </div>
            </section>

            <section className="sources-band">
              <div className="section-heading">
                <Search size={18} />
                <h2>Retrieved tickets</h2>
              </div>
              <div className="source-list">
                {result.retrieval_results.map((ticket) => (
                  <article key={ticket.ticket_id} className="source-item">
                    <div>
                      <strong>{ticket.brand}</strong>
                      <span>{ticket.ticket_id}</span>
                    </div>
                    <p>{ticket.text}</p>
                    <small>Similarity {ticket.score.toFixed(4)}</small>
                  </article>
                ))}
              </div>
            </section>

            <section className="recommendation-band">
              <h2>Deployment recommendation</h2>
              <p>{result.recommendation}</p>
            </section>
          </>
        )}
      </section>
    </main>
  );
}

export default App;
