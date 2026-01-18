"""
Veritas API Router

Endpoints:
- /veritas/analyze - Full analysis
- /veritas/quick-check - Quick myth check
- /veritas/myths - Myths database
- /veritas/search - Search myths
- /veritas/stats - Statistics
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.services.veritas_analyzer import VeritasAnalyzer, get_analyzer
from src.services.output_formatter import (
    format_analysis_markdown,
    format_analysis_html,
)
from src.models.veritas_schema import (
    FullAnalysis,
    FactStatus,
    ContextStatus,
    NarrativeStatus,
    MythCategory,
    HistoricalEra,
)
from src.data.myths_database import get_myths_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/veritas", tags=["Veritas"])


# =============================================================================
# Helper function for enum values
# =============================================================================

def get_enum_value(enum_or_str) -> str:
    """Safely get string value from enum or string."""
    if isinstance(enum_or_str, str):
        return enum_or_str
    return enum_or_str.value if hasattr(enum_or_str, 'value') else str(enum_or_str)


# =============================================================================
# Request/Response Models
# =============================================================================


class AnalyzeRequest(BaseModel):
    """Request for analysis."""
    
    text: str = Field(
        ...,
        min_length=5,
        max_length=10000,
        description="Text to analyze (claim, article, post)",
        examples=["Napoleon was short."],
    )
    deep_analysis: bool = Field(
        default=True,
        description="Deep analysis with context and narrative",
    )
    language: str = Field(
        default="auto",
        description="Language: auto, de, en",
    )
    output_format: str = Field(
        default="json",
        description="Output format: json, markdown, html",
    )


class AnalyzeResponse(BaseModel):
    """Analysis response."""
    
    success: bool
    analysis: Optional[FullAnalysis] = None
    formatted_output: Optional[str] = None
    error: Optional[str] = None


class QuickCheckRequest(BaseModel):
    """Request for Quick Check."""
    
    claim: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Claim to check",
    )


class QuickCheckResponse(BaseModel):
    """Quick Check response."""
    
    found: bool
    myth_id: Optional[str] = None
    claim: Optional[str] = None
    status: Optional[str] = None
    truth: Optional[str] = None
    category: Optional[str] = None
    era: Optional[str] = None
    origin: Optional[str] = None
    popularity: int = 50
    keywords: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    message: Optional[str] = None


class MythResponse(BaseModel):
    """Response for a myth."""
    
    id: str
    claim: str
    claim_en: Optional[str] = None
    category: str
    era: str
    status: str
    truth: str
    truth_en: Optional[str] = None
    origin_source: str
    origin_date: Optional[str] = None
    origin_reason: str
    keywords: list[str]
    popularity: int
    sources: list[dict]
    related_myths: list[str]


class SearchRequest(BaseModel):
    """Search request."""
    
    query: str = Field(..., min_length=2)
    category: Optional[MythCategory] = None
    era: Optional[HistoricalEra] = None
    limit: int = Field(default=10, ge=1, le=50)


class SearchResponse(BaseModel):
    """Search response."""
    
    query: str
    count: int
    results: list[dict]


class StatsResponse(BaseModel):
    """Database statistics."""
    
    total_myths: int
    total_narratives: int
    myths_by_category: dict[str, int]
    myths_by_era: dict[str, int]
    most_popular: list[dict]


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_claim(request: AnalyzeRequest):
    """
    Analyzes a historical claim completely.
    
    Checks:
    - Factual correctness
    - Missing context
    - Narrative patterns
    - Known myths
    
    Returns a detailed verdict with explanation.
    """
    try:
        analyzer = get_analyzer()
        
        analysis = await analyzer.analyze(
            text=request.text,
            deep_analysis=request.deep_analysis,
            language=request.language,
        )
        
        # Formatted output
        formatted = None
        if request.output_format == "markdown":
            formatted = format_analysis_markdown(analysis)
        elif request.output_format == "html":
            formatted = format_analysis_html(analysis)
        
        return AnalyzeResponse(
            success=True,
            analysis=analysis,
            formatted_output=formatted,
        )
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@router.post("/quick-check", response_model=QuickCheckResponse)
async def quick_check(request: QuickCheckRequest):
    """
    Quick check against the myths database.
    
    Only checks if the claim matches a known myth.
    No LLM calls, very fast (<100ms).
    """
    try:
        analyzer = get_analyzer()
        result = await analyzer.quick_check(request.claim)
        
        # Wenn gefunden, hole zusätzliche Daten aus der Datenbank
        if result.get("found") and result.get("myth_id"):
            db = get_myths_database()
            myth = db.myths.get(result["myth_id"])
            
            if myth:
                return QuickCheckResponse(
                    found=True,
                    myth_id=myth.id,
                    claim=myth.claim,
                    status=get_enum_value(myth.status),
                    truth=myth.truth,
                    category=get_enum_value(myth.category),
                    era=get_enum_value(myth.era),
                    origin=myth.origin.source if myth.origin else None,
                    popularity=myth.popularity,
                    keywords=myth.keywords[:10] if myth.keywords else [],
                    sources=[s.title for s in myth.sources[:5]] if myth.sources else [],
                    message=result.get("message"),
                )
        
        # Fallback: Original result
        return QuickCheckResponse(**result)
    
    except Exception as e:
        logger.error(f"Quick check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quick Check failed: {str(e)}",
        )


@router.get("/myths")
async def list_myths(
    category: Optional[MythCategory] = None,
    era: Optional[HistoricalEra] = None,
    limit: int = 50,
):
    """
    Lists myths from the database.
    
    Optionally filterable by category and era.
    Returns object with 'myths' key for consistency.
    """
    db = get_myths_database()
    
    myths = list(db.myths.values())
    
    # Filter
    if category:
        myths = [m for m in myths if get_enum_value(m.category) == get_enum_value(category)]
    if era:
        myths = [m for m in myths if get_enum_value(m.era) == get_enum_value(era)]
    
    # Sort by popularity
    myths.sort(key=lambda m: m.popularity, reverse=True)
    
    # Limit
    myths = myths[:limit]
    
    return {
        "count": len(myths),
        "myths": [
            {
                "id": m.id,
                "claim": m.claim,
                "status": get_enum_value(m.status),
                "truth": m.truth,
                "category": get_enum_value(m.category),
                "era": get_enum_value(m.era),
                "popularity": m.popularity,
                "keywords": m.keywords[:5] if m.keywords else [],
                "related_myths": m.related_myths[:3] if m.related_myths else [],
                "origin": {
                    "date": m.origin.date if m.origin else None,
                    "source": m.origin.source if m.origin else None,
                } if m.origin else None,
            }
            for m in myths
        ]
    }


@router.get("/myths/{myth_id}", response_model=MythResponse)
async def get_myth(myth_id: str):
    """
    Returns detailed information about a myth.
    """
    db = get_myths_database()
    myth = db.get_myth(myth_id)
    
    if not myth:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Myth '{myth_id}' not found",
        )
    
    return MythResponse(
        id=myth.id,
        claim=myth.claim,
        claim_en=myth.claim_en,
        category=get_enum_value(myth.category),
        era=get_enum_value(myth.era),
        status=get_enum_value(myth.status),
        truth=myth.truth,
        truth_en=myth.truth_en,
        origin_source=myth.origin.source,
        origin_date=myth.origin.date,
        origin_reason=myth.origin.reason,
        keywords=myth.keywords,
        popularity=myth.popularity,
        sources=[
            {
                "type": get_enum_value(s.type),
                "title": s.title,
                "author": s.author,
                "year": s.year,
            }
            for s in myth.sources
        ],
        related_myths=myth.related_myths,
    )


@router.post("/search", response_model=SearchResponse)
async def search_myths(request: SearchRequest):
    """
    Searches the myths database.
    """
    db = get_myths_database()
    
    # Text search
    results = db.search_myths(request.query)
    
    # Filter
    if request.category:
        results = [m for m in results if get_enum_value(m.category) == get_enum_value(request.category)]
    if request.era:
        results = [m for m in results if get_enum_value(m.era) == get_enum_value(request.era)]
    
    # Limit
    results = results[:request.limit]
    
    return SearchResponse(
        query=request.query,
        count=len(results),
        results=[
            {
                "id": m.id,
                "claim": m.claim,
                "status": get_enum_value(m.status),
                "truth": m.truth[:200] + "..." if len(m.truth) > 200 else m.truth,
                "category": get_enum_value(m.category),
            }
            for m in results
        ],
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Returns statistics about the database.
    """
    db = get_myths_database()
    
    # Count by category
    by_category = {}
    for myth in db.myths.values():
        cat = get_enum_value(myth.category)
        by_category[cat] = by_category.get(cat, 0) + 1
    
    # Count by era
    by_era = {}
    for myth in db.myths.values():
        era = get_enum_value(myth.era)
        by_era[era] = by_era.get(era, 0) + 1
    
    # Most popular myths
    sorted_myths = sorted(
        db.myths.values(),
        key=lambda m: m.popularity,
        reverse=True
    )[:5]
    
    return StatsResponse(
        total_myths=len(db.myths),
        total_narratives=len(db.narratives),
        myths_by_category=by_category,
        myths_by_era=by_era,
        most_popular=[
            {"id": m.id, "claim": m.claim, "popularity": m.popularity}
            for m in sorted_myths
        ],
    )


@router.get("/narratives")
async def list_narratives():
    """
    Lists known narrative patterns.
    """
    db = get_myths_database()
    
    return [
        {
            "id": n.id,
            "name": n.name,
            "name_en": n.name_en,
            "description": n.description,
            "typical_claims": n.typical_claims,
            "purpose": n.purpose,
        }
        for n in db.narratives.values()
    ]