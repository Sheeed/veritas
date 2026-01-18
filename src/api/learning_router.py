"""
Veritas Learning Router v3.0 - Self-Learning API Endpoints

v3 Upgrades:
- GET /learning/search - BM25 Knowledge Base Search
- GET /learning/training-data?balanced=true - Balanced Training Data

Endpoints:
- POST /learning/mine - Mining starten (parallel)
- GET  /learning/facts/stats - Statistiken
- GET  /learning/facts/sample - Beispiel-Fakten
- GET  /learning/search - BM25 Search (NEU)
- GET  /learning/training-data - Export fuer ML
- POST /learning/reload-kb - Knowledge Base neu laden
- GET  /learning/health - Health Check
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/learning", tags=["Self-Learning"])


# =============================================================================
# Models
# =============================================================================


class MiningResult(BaseModel):
    success: bool
    geographic: int = 0
    capitals: int = 0
    biographical: int = 0
    misconceptions: int = 0
    adversarial: int = 0
    total_fetched: int = 0
    total_new: int = 0
    total_in_db: int = 0
    mining_time_seconds: float = 0.0
    errors: List[str] = []
    debug: dict = {}
    offsets: dict = {}


class StatsResponse(BaseModel):
    total_facts: int
    by_source: dict
    by_type: dict
    by_verdict: dict
    storage_path: str = ""
    offsets: dict = {}


class TrainingItem(BaseModel):
    id: str
    claim: str
    is_true: bool
    explanation: str
    source: str
    claim_type: str
    confidence: float


class SearchResultItem(BaseModel):
    """v3: BM25 Search Result."""

    id: str
    claim: str
    score: float
    is_true: bool
    explanation: str
    source: str
    claim_type: str


class SearchResponse(BaseModel):
    """v3: Search Response."""

    query: str
    total: int
    results: List[SearchResultItem]


# =============================================================================
# Mining Endpoints
# =============================================================================


@router.post("/mine", response_model=MiningResult)
async def mine_facts():
    """
    Startet Fact-Mining aus autoritativen Quellen.

    v3: Parallele Ausfuehrung (4x schneller).
    """
    try:
        from src.ml.self_improver import get_self_improver

        logger.info("Starting parallel fact mining (v3)...")
        system = get_self_improver()
        stats = await system.mine_facts()

        return MiningResult(
            success=True,
            geographic=stats.get("geographic", 0),
            capitals=stats.get("capitals", 0),
            biographical=stats.get("biographical", 0),
            misconceptions=stats.get("misconceptions", 0),
            adversarial=stats.get("adversarial", 0),
            total_fetched=stats.get("total_fetched", 0),
            total_new=stats.get("total_new", 0),
            total_in_db=stats.get("total_in_db", 0),
            mining_time_seconds=stats.get("mining_time_seconds", 0.0),
            errors=stats.get("errors", []),
            debug=stats.get("debug", {}),
            offsets=stats.get("offsets", {}),
        )

    except Exception as e:
        logger.error(f"Mining failed: {e}", exc_info=True)
        return MiningResult(success=False, errors=[str(e)])


@router.get("/facts/stats", response_model=StatsResponse)
async def get_stats():
    """Knowledge Base Statistiken."""
    try:
        from src.ml.self_improver import get_self_improver

        system = get_self_improver()
        stats = system.get_stats()

        return StatsResponse(
            total_facts=stats.get("total_facts", 0),
            by_source=stats.get("by_source", {}),
            by_type=stats.get("by_type", {}),
            by_verdict=stats.get("by_verdict", {}),
            storage_path=stats.get("storage_path", ""),
            offsets=stats.get("offsets", {}),
        )
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        return StatsResponse(total_facts=0, by_source={}, by_type={}, by_verdict={})


@router.get("/facts/sample")
async def get_sample(limit: int = 20, source: Optional[str] = None):
    """Stichprobe der Fakten."""
    try:
        from src.ml.self_improver import get_self_improver

        system = get_self_improver()
        data = system.get_training_data(limit=1000)

        if source:
            data = [d for d in data if d.get("source") == source]

        return {"total": len(data), "sample": data[:limit]}
    except Exception as e:
        return {"error": str(e), "total": 0, "sample": []}


# =============================================================================
# v3: BM25 Search Endpoint
# =============================================================================


@router.get("/search", response_model=SearchResponse)
async def search_kb(
    query: str = Query(..., min_length=2, description="Suchbegriff"),
    top_k: int = Query(default=5, ge=1, le=20, description="Anzahl Ergebnisse"),
    threshold: float = Query(
        default=0.3, ge=0.0, le=1.0, description="Minimaler BM25 Score"
    ),
):
    """
    v3: BM25 Search in der Knowledge Base.

    Durchsucht die lokale Faktenbasis mit BM25 Algorithmus.
    +25% precision gegenueber Jaccard.
    """
    try:
        from src.ml.fact_checker import get_fact_checker

        checker = get_fact_checker()
        results = checker.search_kb(query, top_k=top_k, threshold=threshold)

        return SearchResponse(
            query=query,
            total=len(results),
            results=[
                SearchResultItem(
                    id=r.id,
                    claim=r.claim,
                    score=r.score,
                    is_true=r.is_true,
                    explanation=r.explanation,
                    source=r.source,
                    claim_type=r.claim_type,
                )
                for r in results
            ],
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Training Data
# =============================================================================


@router.get("/training-data")
async def get_training_data(
    limit: int = Query(default=1000, ge=1, le=10000),
    claim_type: Optional[str] = None,
    only_true: bool = False,
    only_false: bool = False,
    balanced: bool = Query(default=False, description="v3: 50% TRUE, 50% FALSE"),
):
    """
    Trainingsdaten fuer ML.

    v3: balanced=true fuer ausgewogene Daten.
    """
    try:
        from src.ml.self_improver import get_self_improver

        system = get_self_improver()
        data = system.get_training_data(limit=10000, balanced=balanced)

        if claim_type:
            data = [d for d in data if d.get("claim_type") == claim_type]
        if only_true:
            data = [d for d in data if d.get("is_true")]
        if only_false:
            data = [d for d in data if not d.get("is_true")]

        return {
            "total": len(data),
            "balanced": balanced,
            "items": [TrainingItem(**d) for d in data[:limit]],
        }
    except Exception as e:
        return {"error": str(e), "total": 0, "items": []}


# =============================================================================
# Health & Utils
# =============================================================================


@router.get("/health")
async def health():
    """Self-Learning System Status."""
    try:
        from src.ml.self_improver import get_self_improver
        from src.ml.fact_checker import get_fact_checker

        system = get_self_improver()
        checker = get_fact_checker()

        stats = system.get_stats()
        cache_stats = checker.get_cache_stats()

        return {
            "status": "ok",
            "version": "3.0",
            "total_facts": stats.get("total_facts", 0),
            "sources": list(stats.get("by_source", {}).keys()),
            "cache": cache_stats.get("cache", {}),
            "bm25": cache_stats.get("local_kb", {}).get("bm25", {}),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/reload-kb")
async def reload_kb():
    """Laedt Knowledge Base im Fact Checker neu."""
    try:
        from src.ml.fact_checker import get_fact_checker

        checker = get_fact_checker()
        checker.reload_kb()

        stats = checker.get_cache_stats()

        return {
            "success": True,
            "message": "Knowledge Base reloaded with BM25 index",
            "facts": stats.get("local_kb", {}).get("facts", 0),
            "bm25": stats.get("local_kb", {}).get("bm25", {}),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/test-wikidata")
async def test_wikidata():
    """Testet Wikidata-Verbindung."""
    import httpx

    sparql = """
    SELECT ?countryLabel WHERE {
        ?country wdt:P31 wd:Q6256 .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "de". }
    } LIMIT 5
    """

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://query.wikidata.org/sparql",
                params={"query": sparql, "format": "json"},
                headers={"User-Agent": "Veritas-Test/3.0"},
            )

            if response.status_code != 200:
                return {"success": False, "status": response.status_code}

            results = response.json().get("results", {}).get("bindings", [])
            countries = [r.get("countryLabel", {}).get("value", "") for r in results]

            return {"success": True, "results_count": len(results), "sample": countries}
    except Exception as e:
        return {"success": False, "error": str(e)}
