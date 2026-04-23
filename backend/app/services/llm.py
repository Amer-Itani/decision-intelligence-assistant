"""LLM provider adapter with Groq implementation and local fallback."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.core.config import Settings
from app.schemas.analysis import GeneratedAnswer, LlmPrediction, RetrievedTicket
from app.services.features import weak_label_priority


class ZeroShotPriority(BaseModel):
    """Validated LLM priority classification output."""

    label: str = Field(pattern="^(low|normal|high|urgent)$")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class LlmService:
    """Generate RAG/non-RAG answers and zero-shot priority predictions."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None
        if settings.groq_api_key:
            from groq import Groq

            self._client = Groq(api_key=settings.groq_api_key)

    def generate_rag_answer(
        self,
        ticket_text: str,
        retrieved_tickets: list[RetrievedTicket],
    ) -> GeneratedAnswer:
        """Generate an answer grounded in retrieved tickets."""

        if not self._client:
            return self._fallback_rag_answer(ticket_text, retrieved_tickets)

        context = "\n".join(
            f"- [{ticket.ticket_id}, score={ticket.score}] {ticket.text}"
            for ticket in retrieved_tickets
        )
        prompt = (
            "You are a customer-support decision assistant. Answer only using "
            "the retrieved historical cases. If evidence is weak, say that.\n\n"
            f"Ticket:\n{ticket_text}\n\nRetrieved cases:\n{context}"
        )
        return self._chat_answer(prompt, "rag")

    def generate_non_rag_answer(self, ticket_text: str) -> GeneratedAnswer:
        """Generate an answer without retrieved context."""

        if not self._client:
            return self._fallback_non_rag_answer(ticket_text)

        prompt = (
            "You are a customer-support assistant. Suggest a concise next action "
            "for the customer-support team based only on this ticket:\n\n"
            f"{ticket_text}"
        )
        return self._chat_answer(prompt, "non-rag")

    def classify_priority(self, ticket_text: str) -> LlmPrediction:
        """Ask the LLM for zero-shot priority classification."""

        start_time = time.perf_counter()
        if not self._client:
            label = weak_label_priority(ticket_text)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return LlmPrediction(
                label=label,
                confidence=0.68,
                reason="Local fallback used because no GROQ_API_KEY is configured.",
                latency_ms=round(latency_ms, 2),
                cost_usd=0.0,
            )

        prompt = (
            "Classify this support ticket priority as exactly one of: low, "
            "normal, high, urgent. Return JSON only with keys label, confidence, "
            "reason.\n\n"
            f"Ticket: {ticket_text}"
        )
        response_text, latency_ms, cost_usd = self._chat_text(prompt)
        parsed = self._parse_priority_response(response_text, ticket_text)

        return LlmPrediction(
            label=parsed.label,
            confidence=parsed.confidence,
            reason=parsed.reason,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )

    def _chat_answer(self, prompt: str, mode: str) -> GeneratedAnswer:
        response_text, latency_ms, cost_usd = self._chat_text(prompt)
        confidence_note = (
            "Grounded in retrieved cases." if mode == "rag" else "No retrieval context used."
        )
        return GeneratedAnswer(
            answer=response_text,
            confidence_note=confidence_note,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )

    def _chat_text(self, prompt: str) -> tuple[str, float, float]:
        start_time = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        content = response.choices[0].message.content or ""
        cost_usd = self._estimate_cost(response)
        return content.strip(), round(latency_ms, 2), cost_usd

    def _estimate_cost(self, response: Any) -> float:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0.0
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = (
            input_tokens * self._settings.llm_input_cost_per_1m_tokens
            + output_tokens * self._settings.llm_output_cost_per_1m_tokens
        ) / 1_000_000
        return round(cost, 6)

    def _parse_priority_response(
        self,
        response_text: str,
        ticket_text: str,
    ) -> ZeroShotPriority:
        try:
            return ZeroShotPriority.model_validate_json(response_text)
        except ValidationError:
            pass
        except ValueError:
            pass

        try:
            start_index = response_text.index("{")
            end_index = response_text.rindex("}") + 1
            payload = json.loads(response_text[start_index:end_index])
            return ZeroShotPriority.model_validate(payload)
        except (ValueError, ValidationError, json.JSONDecodeError):
            return ZeroShotPriority(
                label=weak_label_priority(ticket_text),
                confidence=0.5,
                reason="LLM response could not be validated; rule fallback applied.",
            )

    @staticmethod
    def _fallback_rag_answer(
        ticket_text: str,
        retrieved_tickets: list[RetrievedTicket],
    ) -> GeneratedAnswer:
        start_time = time.perf_counter()
        if retrieved_tickets:
            best = retrieved_tickets[0]
            answer = (
                "Similar historical cases suggest acknowledging the issue, "
                "asking for account or device details, and routing the ticket to "
                f"the relevant support queue. Closest case: {best.text[:220]}"
            )
            note = f"Fallback answer based on top retrieved score {best.score}."
        else:
            answer = (
                "No strong historical context was available. A safe next step is "
                "to acknowledge the problem, ask for missing details, and escalate "
                "if the customer is blocked."
            )
            note = "Fallback answer without retrieved evidence."
        latency_ms = (time.perf_counter() - start_time) * 1000
        return GeneratedAnswer(
            answer=answer,
            confidence_note=note,
            latency_ms=round(latency_ms, 2),
            cost_usd=0.0,
        )

    @staticmethod
    def _fallback_non_rag_answer(ticket_text: str) -> GeneratedAnswer:
        start_time = time.perf_counter()
        label = weak_label_priority(ticket_text)
        answer = (
            "Acknowledge the customer's issue, confirm key details, and provide a "
            f"clear next step. The ticket appears to be {label} priority based on "
            "surface-level language signals."
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        return GeneratedAnswer(
            answer=answer,
            confidence_note="Local fallback used because no GROQ_API_KEY is configured.",
            latency_ms=round(latency_ms, 2),
            cost_usd=0.0,
        )
