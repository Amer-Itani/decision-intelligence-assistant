"""Application orchestration for the four-way comparison endpoint."""

from __future__ import annotations

from uuid import uuid4

from app.core.config import Settings
from app.schemas.analysis import AnalyzeResponse
from app.services.llm import LlmService
from app.services.ml import PriorityModelService
from app.services.query_logger import QueryLogger
from app.services.retrieval import RetrievalService


class AnalysisService:
    """Coordinate retrieval, generation, ML prediction, and logging."""

    def __init__(self, settings: Settings) -> None:
        self._retrieval = RetrievalService(settings)
        self._llm = LlmService(settings)
        self._ml = PriorityModelService(settings)
        self._logger = QueryLogger(settings)

    def analyze(self, ticket_text: str, top_k: int) -> AnalyzeResponse:
        """Run the complete comparison workflow for one ticket."""

        request_id = str(uuid4())
        retrieval_results, retrieval_latency_ms = self._retrieval.search(
            query=ticket_text,
            top_k=top_k,
        )
        rag_answer = self._llm.generate_rag_answer(ticket_text, retrieval_results)
        non_rag_answer = self._llm.generate_non_rag_answer(ticket_text)
        ml_prediction = self._ml.predict(ticket_text)
        llm_prediction = self._llm.classify_priority(ticket_text)
        recommendation = self._build_recommendation(
            ml_label=ml_prediction.label,
            llm_label=llm_prediction.label,
            retrieval_count=len(retrieval_results),
        )

        response = AnalyzeResponse(
            request_id=request_id,
            ticket_text=ticket_text,
            retrieval_results=retrieval_results,
            rag_answer=rag_answer,
            non_rag_answer=non_rag_answer,
            ml_prediction=ml_prediction,
            llm_prediction=llm_prediction,
            recommendation=recommendation,
        )
        self._logger.write(
            {
                "request_id": request_id,
                "ticket_text": ticket_text,
                "retrieval_latency_ms": retrieval_latency_ms,
                "response": response,
            }
        )
        return response

    @staticmethod
    def _build_recommendation(
        ml_label: str,
        llm_label: str,
        retrieval_count: int,
    ) -> str:
        if ml_label == llm_label:
            return (
                "Both priority methods agree. In production, use the ML model "
                "for high-volume triage and reserve the LLM for disputed or "
                "high-risk tickets."
            )
        if retrieval_count == 0:
            return (
                "Priority methods disagree and retrieval evidence is weak. Route "
                "the ticket for human review or an LLM second pass."
            )
        return (
            "Priority methods disagree. Use the ML result for fast queueing, but "
            "surface the LLM reason and retrieved examples for reviewer context."
        )
