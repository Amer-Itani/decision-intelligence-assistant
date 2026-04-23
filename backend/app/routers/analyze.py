"""Ticket analysis API route."""

from functools import lru_cache

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.services.analysis import AnalysisService


router = APIRouter(tags=["analysis"])


@lru_cache
def get_analysis_service() -> AnalysisService:
    """Create a cached analysis service for API requests."""

    return AnalysisService(get_settings())


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_ticket(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run RAG, non-RAG, ML, and LLM priority analysis."""

    service = get_analysis_service()
    return service.analyze(
        ticket_text=request.ticket_text,
        top_k=request.top_k,
    )
