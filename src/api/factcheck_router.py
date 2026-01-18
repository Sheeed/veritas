"""
Veritas Factcheck Router - Fact Checking API Endpoints

Endpoints:
- POST /factcheck/check - Vollstaendiger Faktencheck
- POST /factcheck/quick - Schneller Check (nur Verdict)
- GET  /factcheck/health - System Status
- GET  /factcheck/cache/stats - Cache Statistiken
- POST /factcheck/cache/clear - Cache leeren
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/factcheck", tags=["Fact Check"])


# =============================================================================
# Models
# =============================================================================


class FactCheckRequest(BaseModel):
    """Anfrage fuer Faktencheck."""

    claim: str = Field(..., min_length=3, description="Die zu pruefende Behauptung")
    skip_cache: bool = Field(default=False, description="Cache umgehen")


class EvidenceResponse(BaseModel):
    """Evidenz aus einer Quelle."""

    source: str
    content: str
    supports_claim: bool
    confidence: float
    url: Optional[str] = None


class FactCheckResponse(BaseModel):
    """Vollstaendige Faktencheck-Antwort."""

    claim: str
    verdict: str
    verdict_label: str
    confidence: float
    explanation: str
    correction: Optional[str] = None
    claim_type: str
    evidence: List[EvidenceResponse] = []
    sources_checked: int = 0
    llm_used: bool = False
    wikidata_used: bool = False
    wikipedia_used: bool = False
    local_kb_used: bool = False
    cached: bool = False
    processing_time_ms: int = 0


class QuickCheckResponse(BaseModel):
    """Schnelle Faktencheck-Antwort (nur Verdict)."""

    claim: str
    verdict: str
    verdict_label: str
    confidence: float
    processing_time_ms: int = 0


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/check", response_model=FactCheckResponse)
async def check_claim(request: FactCheckRequest):
    """
    Vollstaendiger Faktencheck.

    Ablauf:
    1. Cache pruefen
    2. Lokale Knowledge Base durchsuchen
    3. Parallel: LLM + Wikidata + Wikipedia
    4. Weighted Voting

    Returns:
        Verdict, Confidence, Erklaerung, Evidenz
    """
    try:
        from src.ml.fact_checker import get_fact_checker

        checker = get_fact_checker()
        result = await checker.check(request.claim, skip_cache=request.skip_cache)

        return FactCheckResponse(
            claim=result.claim,
            verdict=result.verdict.value,
            verdict_label=result.verdict_label,
            confidence=result.confidence,
            explanation=result.explanation,
            correction=result.correction,
            claim_type=result.claim_type.value,
            evidence=[
                EvidenceResponse(
                    source=e.source,
                    content=e.content,
                    supports_claim=e.supports_claim,
                    confidence=e.confidence,
                    url=e.url,
                )
                for e in result.evidence
            ],
            sources_checked=result.sources_checked,
            llm_used=result.llm_used,
            wikidata_used=result.wikidata_used,
            wikipedia_used=result.wikipedia_used,
            local_kb_used=result.local_kb_used,
            cached=result.cached,
            processing_time_ms=result.processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Fact check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick", response_model=QuickCheckResponse)
async def quick_check(request: FactCheckRequest):
    """
    Schneller Faktencheck - nur Verdict.

    Gleicher Ablauf wie /check, aber kompakte Antwort.
    """
    try:
        from src.ml.fact_checker import get_fact_checker

        checker = get_fact_checker()
        result = await checker.check(request.claim, skip_cache=request.skip_cache)

        return QuickCheckResponse(
            claim=result.claim,
            verdict=result.verdict.value,
            verdict_label=result.verdict_label,
            confidence=result.confidence,
            processing_time_ms=result.processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Quick check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """Fact Checker Status."""
    import os

    try:
        from src.ml.fact_checker import get_fact_checker

        checker = get_fact_checker()
        stats = checker.get_cache_stats()

        return {
            "status": "ok",
            "llm_available": bool(os.getenv("GROQ_API_KEY")),
            "wikidata_available": True,
            "cache": stats.get("cache", {}),
            "local_kb": stats.get("local_kb", {}),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/cache/stats")
async def cache_stats():
    """Cache Statistiken."""
    try:
        from src.ml.fact_checker import get_fact_checker

        checker = get_fact_checker()
        return checker.get_cache_stats()
    except Exception as e:
        return {"error": str(e)}


@router.post("/cache/clear")
async def clear_cache():
    """Cache leeren."""
    try:
        from src.ml.fact_checker import get_fact_checker

        checker = get_fact_checker()
        checker.cache.clear()

        return {"success": True, "message": "Cache cleared"}
    except Exception as e:
        return {"success": False, "error": str(e)}
