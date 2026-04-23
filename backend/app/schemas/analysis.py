"""Request and response schemas for ticket analysis."""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Payload submitted by the frontend for one ticket analysis."""

    ticket_text: str = Field(..., min_length=3, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=10)


class RetrievedTicket(BaseModel):
    """One historical ticket returned by retrieval."""

    ticket_id: str
    brand: str
    text: str
    score: float
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class GeneratedAnswer(BaseModel):
    """LLM or fallback answer with operational measurements."""

    answer: str
    confidence_note: str | None = None
    latency_ms: float
    cost_usd: float


class MlPrediction(BaseModel):
    """Priority prediction from the trained classifier."""

    label: str
    confidence: float
    model_name: str
    latency_ms: float
    cost_usd: float = 0.0


class LlmPrediction(BaseModel):
    """Priority prediction from the LLM zero-shot classifier."""

    label: str
    confidence: float
    reason: str
    latency_ms: float
    cost_usd: float


class AnalyzeResponse(BaseModel):
    """Complete four-way comparison returned by the backend."""

    request_id: str
    ticket_text: str
    retrieval_results: list[RetrievedTicket]
    rag_answer: GeneratedAnswer
    non_rag_answer: GeneratedAnswer
    ml_prediction: MlPrediction
    llm_prediction: LlmPrediction
    recommendation: str
