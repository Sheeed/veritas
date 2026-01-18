"""
Veritas - Neo4j Graph API Router

Endpoints für Graph-basierte Abfragen und Analysen.
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["Graph"])


# =============================================================================
# Response Models
# =============================================================================

class GraphStatsResponse(BaseModel):
    """Graph statistics response."""
    connected: bool
    total_myths: int = 0
    total_narratives: int = 0
    total_persons: int = 0
    total_sources: int = 0
    total_relationships: int = 0


class RelatedMythResponse(BaseModel):
    """Related myth with distance."""
    id: str
    claim: str
    claim_en: Optional[str] = None
    status: str
    distance: int


class ImportResponse(BaseModel):
    """Import result."""
    success: bool
    myths_imported: int = 0
    narratives_imported: int = 0
    message: str = ""


class ClusterResponse(BaseModel):
    """Myth cluster."""
    myth_id: str
    claim: str
    related_ids: List[str]
    connections: int


# =============================================================================
# Helper
# =============================================================================

def get_graph_service():
    """Gets the graph service, handling import errors gracefully."""
    try:
        from src.services.neo4j_graph_service import get_graph_service as _get_service
        return _get_service()
    except ImportError as e:
        logger.warning(f"Neo4j graph service not available: {e}")
        return None


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/status", response_model=GraphStatsResponse)
async def get_graph_status():
    """
    Returns Neo4j connection status and statistics.
    """
    service = get_graph_service()
    
    if not service:
        return GraphStatsResponse(connected=False)
    
    if not service.ensure_connected():
        return GraphStatsResponse(connected=False)
    
    stats = service.get_graph_stats()
    return GraphStatsResponse(**stats)


@router.post("/import", response_model=ImportResponse)
async def import_to_graph():
    """
    Imports all myths and narratives from database to Neo4j graph.
    
    This will:
    1. Create schema (indexes, constraints)
    2. Import all myths as nodes
    3. Import all narratives as nodes
    4. Create relationships between them
    """
    service = get_graph_service()
    
    if not service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j graph service not available"
        )
    
    if not service.ensure_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cannot connect to Neo4j. Make sure it's running."
        )
    
    try:
        result = service.import_all_from_database()
        return ImportResponse(
            success=True,
            myths_imported=result["myths_imported"],
            narratives_imported=result["narratives_imported"],
            message="Import successful"
        )
    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Import failed: {str(e)}"
        )


@router.get("/myths/{myth_id}/related", response_model=List[dict])
async def get_related_myths(myth_id: str, depth: int = 2):
    """
    Finds related myths through graph traversal.
    
    Parameters:
    - myth_id: ID of the myth to find relations for
    - depth: How many hops to traverse (1-3)
    """
    if depth < 1 or depth > 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Depth must be between 1 and 3"
        )
    
    service = get_graph_service()
    
    if not service or not service.ensure_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available"
        )
    
    results = service.get_related_myths(myth_id, depth)
    return results


@router.get("/narratives/{narrative_id}/myths", response_model=List[dict])
async def get_myths_by_narrative(narrative_id: str):
    """
    Returns all myths belonging to a narrative pattern.
    """
    service = get_graph_service()
    
    if not service or not service.ensure_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available"
        )
    
    results = service.get_myths_by_narrative(narrative_id)
    return results


@router.get("/debunkers/{name}/myths", response_model=List[dict])
async def get_myths_by_debunker(name: str):
    """
    Returns all myths debunked by a specific person.
    """
    service = get_graph_service()
    
    if not service or not service.ensure_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available"
        )
    
    results = service.get_myths_by_debunker(name)
    return results


@router.get("/search")
async def search_graph(query: str, limit: int = 10):
    """
    Full-text search across the myth graph.
    """
    if len(query) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be at least 2 characters"
        )
    
    service = get_graph_service()
    
    if not service or not service.ensure_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available"
        )
    
    results = service.search_myths_fulltext(query, limit)
    return results


@router.get("/clusters")
async def get_myth_clusters():
    """
    Finds clusters of related myths.
    
    Returns myths with the most connections to other myths.
    """
    service = get_graph_service()
    
    if not service or not service.ensure_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available"
        )
    
    clusters = service.find_myth_clusters()
    return clusters


@router.get("/categories", response_model=dict)
async def get_category_distribution():
    """
    Returns distribution of myths by category from the graph.
    """
    service = get_graph_service()
    
    if not service or not service.ensure_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available"
        )
    
    return service.get_category_distribution()


@router.delete("/clear")
async def clear_graph():
    """
    Clears all data from Neo4j (DANGEROUS!).
    
    Use with caution - this deletes all nodes and relationships.
    """
    service = get_graph_service()
    
    if not service or not service.ensure_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available"
        )
    
    try:
        service.clear_database()
        return {"success": True, "message": "Graph cleared"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear: {str(e)}"
        )