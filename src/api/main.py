"""
FastAPI Application für The History Guardian.

Bietet REST-Endpoints für:
- Knowledge Graph Extraktion aus Text
- Claim-Ingestion in Neo4j
- Verifikation gegen Ground Truth
- Batch-Verarbeitung
- ML Confidence Scoring
- Datenquellen-Import
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, status, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.agents.extraction import ExtractionAgent
from src.config import get_settings
from src.db.graph_db import GraphManager, get_graph_manager
from src.models.schema import KnowledgeGraphExtraction, NodeType

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class ExtractionRequest(BaseModel):
    """Request für Knowledge Graph Extraktion."""

    text: str = Field(
        ...,
        min_length=10,
        max_length=50000,
        description="Text zur Extraktion",
        examples=[
            "Napoleon Bonaparte wurde am 15. August 1769 auf Korsika geboren."
        ],
    )
    use_few_shot: bool = Field(
        default=True,
        description="Few-Shot Beispiele für bessere Ergebnisse verwenden",
    )


class ExtractionResponse(BaseModel):
    """Response mit extrahiertem Knowledge Graph."""

    success: bool
    extraction: KnowledgeGraphExtraction
    message: str = ""


class IngestRequest(BaseModel):
    """Request für Graph-Ingestion."""

    text: str = Field(..., min_length=10, max_length=50000)
    as_fact: bool = Field(
        default=False,
        description="Als verifizierter Fakt (True) oder als Claim (False) speichern",
    )


class IngestResponse(BaseModel):
    """Response nach Ingestion."""

    success: bool
    nodes_added: int
    relationships_added: int
    message: str = ""


class HealthResponse(BaseModel):
    """Health Check Response."""

    status: str
    neo4j_connected: bool
    version: str = "0.1.0"


class StatsResponse(BaseModel):
    """Graph Statistics Response."""

    success: bool
    statistics: dict[str, Any]


# =============================================================================
# Application Lifecycle
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application Lifecycle Management."""
    logger.info("Starting History Guardian API...")
    
    # Startup: GraphManager initialisieren
    try:
        graph_manager = await get_graph_manager()
        app.state.graph_manager = graph_manager
        app.state.extraction_agent = ExtractionAgent()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown: Verbindungen schließen
    logger.info("Shutting down History Guardian API...")
    if hasattr(app.state, "graph_manager"):
        await app.state.graph_manager.disconnect()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="The History Guardian",
    description="""
    🛡️ **GraphRAG-basiertes System zur Erkennung von Fake News und historischen Ungenauigkeiten.**
    
    ## Features
    
    * **Knowledge Graph Extraktion**: Extrahiert Entitäten und Beziehungen aus Text
    * **Claim Ingestion**: Speichert Behauptungen zur späteren Verifikation
    * **Fact Database**: Verifizierte historische Fakten als Ground Truth
    * **Verification**: Prüft Claims gegen die Faktendatenbank
    
    ## Workflow
    
    1. Text mit Behauptungen an `/extract` senden
    2. Extrahierten Graph mit `/ingest` speichern
    3. Mit `/verify` gegen Ground Truth prüfen
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# CORS für Frontend-Zugriff
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion einschränken!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/", tags=["Info"])
async def root():
    """Willkommensnachricht."""
    return {
        "name": "The History Guardian",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health Check Endpoint."""
    neo4j_connected = False
    
    try:
        graph_manager: GraphManager = app.state.graph_manager
        async with graph_manager.session() as session:
            await session.run("RETURN 1")
            neo4j_connected = True
    except Exception as e:
        logger.warning(f"Neo4j health check failed: {e}")
    
    return HealthResponse(
        status="healthy" if neo4j_connected else "degraded",
        neo4j_connected=neo4j_connected,
    )


@app.post("/extract", response_model=ExtractionResponse, tags=["Extraction"])
async def extract_knowledge_graph(request: ExtractionRequest):
    """
    Extrahiert einen Knowledge Graph aus dem gegebenen Text.
    
    Verwendet GPT-4o mit strukturiertem Output für konsistente Ergebnisse.
    """
    try:
        agent: ExtractionAgent = app.state.extraction_agent
        extraction = await agent.extract_knowledge_graph(
            text=request.text,
            use_few_shot=request.use_few_shot,
        )
        
        return ExtractionResponse(
            success=True,
            extraction=extraction,
            message=f"Extracted {len(extraction.nodes)} nodes and {len(extraction.relationships)} relationships",
        )
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}",
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text(request: IngestRequest):
    """
    Extrahiert und speichert einen Knowledge Graph in Neo4j.
    
    - `as_fact=False` (default): Speichert als :Claim zur Verifikation
    - `as_fact=True`: Speichert als :Fact (verifizierte Ground Truth)
    """
    try:
        # Extraktion
        agent: ExtractionAgent = app.state.extraction_agent
        extraction = await agent.extract_knowledge_graph(
            text=request.text,
            mark_as_claim=not request.as_fact,
        )
        
        if not extraction.nodes and not extraction.relationships:
            return IngestResponse(
                success=False,
                nodes_added=0,
                relationships_added=0,
                message="No entities or relationships could be extracted from the text",
            )
        
        # Ingestion
        graph_manager: GraphManager = app.state.graph_manager
        
        if request.as_fact:
            stats = await graph_manager.add_fact_graph(extraction)
        else:
            stats = await graph_manager.add_claim_graph(extraction)
        
        return IngestResponse(
            success=True,
            nodes_added=stats["nodes_added"],
            relationships_added=stats["relationships_added"],
            message=f"Successfully ingested as {'Fact' if request.as_fact else 'Claim'}",
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@app.get("/stats", response_model=StatsResponse, tags=["Info"])
async def get_graph_statistics():
    """Gibt Statistiken über den Knowledge Graph zurück."""
    try:
        graph_manager: GraphManager = app.state.graph_manager
        stats = await graph_manager.get_statistics()
        
        return StatsResponse(success=True, statistics=stats)
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.delete("/claims", tags=["Management"])
async def clear_all_claims():
    """
    Löscht alle Claim-Nodes aus der Datenbank.
    
    ⚠️ Vorsicht: Diese Aktion kann nicht rückgängig gemacht werden!
    """
    try:
        graph_manager: GraphManager = app.state.graph_manager
        deleted = await graph_manager.clear_claims()
        
        return {"success": True, "deleted_nodes": deleted}
        
    except Exception as e:
        logger.error(f"Failed to clear claims: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# =============================================================================
# Extended Request/Response Models
# =============================================================================


class ValidationRequest(BaseModel):
    """Request für Claim-Validierung."""
    extraction: KnowledgeGraphExtraction


class ValidationResponse(BaseModel):
    """Response mit Validierungsergebnis."""
    success: bool
    overall_status: str
    overall_confidence: float
    summary: str
    recommendation: str
    all_issues: list[dict[str, Any]] = Field(default_factory=list)
    entity_matches: list[dict[str, Any]] = Field(default_factory=list)


class BatchRequest(BaseModel):
    """Request für Batch-Verarbeitung."""
    texts: list[str] = Field(..., min_length=1, max_length=1000)
    max_concurrent: int = Field(default=5, ge=1, le=20)
    as_facts: bool = Field(default=False)


class BatchStatusResponse(BaseModel):
    """Status eines Batch-Jobs."""
    job_id: str
    status: str
    progress_percent: float
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int


class DataSourceImportRequest(BaseModel):
    """Request für Datenquellen-Import."""
    query: str = Field(..., min_length=2, max_length=200)
    entity_type: str = Field(..., description="Person, Event, Location, Organization")
    sources: list[str] = Field(default=["wikidata", "dbpedia"])


class ConfidenceScoreRequest(BaseModel):
    """Request für ML Confidence Scoring."""
    extraction: KnowledgeGraphExtraction
    source_text: str | None = None


class ConfidenceScoreResponse(BaseModel):
    """Response mit ML Confidence Score."""
    confidence: float
    components: dict[str, float]
    explanation: dict[str, Any] | None = None
    is_trained: bool


# =============================================================================
# Validation Endpoints
# =============================================================================


@app.post("/validate", response_model=ValidationResponse, tags=["Validation"])
async def validate_claim(request: ValidationRequest):
    """
    Validiert eine Extraktion gegen die Ground Truth Datenbank.
    
    Prüft:
    - Entity Resolution (Sind die Entitäten bekannt?)
    - Chronologische Konsistenz (Sind die Daten plausibel?)
    - Beziehungsvalidierung (Stimmen die Beziehungen?)
    """
    try:
        # Lazy import to avoid circular dependencies
        from src.validation.validator import ClaimValidator
        
        graph_manager: GraphManager = app.state.graph_manager
        validator = ClaimValidator(graph_manager)
        
        result = await validator.validate_extraction(request.extraction)
        
        return ValidationResponse(
            success=True,
            overall_status=result.overall_status.value,
            overall_confidence=result.overall_confidence,
            summary=result.summary,
            recommendation=result.recommendation,
            all_issues=[issue.model_dump() for issue in result.all_issues],
            entity_matches=[match.model_dump() for match in result.entity_matches],
        )
        
    except ImportError:
        # Validation module not yet implemented
        return ValidationResponse(
            success=False,
            overall_status="unverifiable",
            overall_confidence=0.5,
            summary="Validation module not available",
            recommendation="Install validation dependencies",
        )
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        )


# =============================================================================
# Batch Processing Endpoints
# =============================================================================

# In-memory job storage (use Redis/DB in production)
_batch_jobs: dict[str, Any] = {}


@app.post("/batch/start", tags=["Batch Processing"])
async def start_batch_job(request: BatchRequest, background_tasks: BackgroundTasks):
    """
    Startet einen Batch-Verarbeitungsjob.
    
    Verarbeitet mehrere Texte parallel mit konfigurierbarer Concurrency.
    """
    try:
        from src.processing.batch import BatchProcessor, BatchJobConfig
        
        processor = BatchProcessor(
            extraction_agent=app.state.extraction_agent,
            graph_manager=app.state.graph_manager,
        )
        
        config = BatchJobConfig(
            max_concurrent=request.max_concurrent,
            as_facts=request.as_facts,
        )
        
        job = processor.create_job(config)
        job.add_texts(request.texts)
        
        # Store job reference
        job_id = str(job.id)
        _batch_jobs[job_id] = {"job": job, "processor": processor}
        
        # Process in background
        async def process_job():
            try:
                await processor.process_job(job)
            except Exception as e:
                logger.error(f"Batch job {job_id} failed: {e}")
        
        background_tasks.add_task(process_job)
        
        return {
            "success": True,
            "job_id": job_id,
            "total_items": len(request.texts),
            "message": "Batch job started",
        }
        
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Batch processing module not available",
        )
    except Exception as e:
        logger.error(f"Failed to start batch job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/batch/{job_id}/status", response_model=BatchStatusResponse, tags=["Batch Processing"])
async def get_batch_status(job_id: str):
    """Gibt den Status eines Batch-Jobs zurück."""
    if job_id not in _batch_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    
    job = _batch_jobs[job_id]["job"]
    progress = job.progress
    
    return BatchStatusResponse(
        job_id=job_id,
        status=progress.status.value,
        progress_percent=progress.progress_percent,
        total_items=progress.total_items,
        processed_items=progress.processed_items,
        successful_items=progress.successful_items,
        failed_items=progress.failed_items,
    )


# =============================================================================
# Data Source Import Endpoints
# =============================================================================


@app.post("/import/external", tags=["Data Sources"])
async def import_from_external_sources(request: DataSourceImportRequest):
    """
    Importiert Daten aus externen Quellen (Wikidata, DBpedia).
    
    Die importierten Daten werden als verifizierte Fakten gespeichert.
    """
    try:
        from src.datasources.external import DataSourceManager, DataSourceType
        
        # Map string to NodeType
        type_mapping = {
            "person": NodeType.PERSON,
            "event": NodeType.EVENT,
            "location": NodeType.LOCATION,
            "organization": NodeType.ORGANIZATION,
        }
        
        entity_type = type_mapping.get(request.entity_type.lower())
        if not entity_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid entity type: {request.entity_type}",
            )
        
        manager = DataSourceManager()
        manager.register_defaults()
        
        # Filter sources
        source_mapping = {
            "wikidata": DataSourceType.WIKIDATA,
            "dbpedia": DataSourceType.DBPEDIA,
        }
        sources = [source_mapping[s] for s in request.sources if s in source_mapping]
        
        results = await manager.search_all(request.query, entity_type, sources)
        
        # Merge and store
        merged = manager.merge_extractions(results)
        
        graph_manager: GraphManager = app.state.graph_manager
        stats = await graph_manager.add_fact_graph(merged)
        
        await manager.close_all()
        
        return {
            "success": True,
            "sources_queried": len(results),
            "nodes_imported": stats["nodes_added"],
            "relationships_imported": stats["relationships_added"],
        }
        
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Data sources module not available",
        )
    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# =============================================================================
# ML Confidence Scoring Endpoints
# =============================================================================


@app.post("/score/confidence", response_model=ConfidenceScoreResponse, tags=["ML Scoring"])
async def score_confidence(request: ConfidenceScoreRequest):
    """
    Berechnet einen ML-basierten Konfidenz-Score für eine Extraktion.
    
    Verwendet ein Ensemble aus:
    - Regelbasiertem Scoring
    - Logistischer Regression
    - Random Forest
    """
    try:
        from src.ml.confidence import EnsembleConfidenceScorer
        
        scorer = EnsembleConfidenceScorer()
        result = scorer.score_extraction(
            extraction=request.extraction,
            source_text=request.source_text,
        )
        
        return ConfidenceScoreResponse(
            confidence=result["confidence"],
            components=result["components"],
            explanation=result.get("explanation"),
            is_trained=result["is_trained"],
        )
        
    except ImportError:
        # Return rule-based fallback
        return ConfidenceScoreResponse(
            confidence=0.5,
            components={"rule_based": 0.5},
            explanation={"message": "ML module not available, using fallback"},
            is_trained=False,
        )
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# =============================================================================
# Authority Data Sources (Bibliografische Normdateien)
# =============================================================================


class AuthorityImportRequest(BaseModel):
    """Request für Import aus autoritativen Quellen."""
    query: str = Field(..., min_length=2, max_length=200)
    entity_type: str = Field(..., description="Person, Event, Location, Organization")
    sources: list[str] = Field(
        default=["gnd", "viaf", "loc"],
        description="Autoritative Quellen: gnd, viaf, loc, getty_tgn, getty_ulan"
    )


@app.post("/import/authority", tags=["Authority Sources"])
async def import_from_authority_sources(request: AuthorityImportRequest):
    """
    Importiert verifizierte Daten aus bibliothekarischen Normdateien.
    
    **Unterstützte Quellen (KEINE crowdsourced Daten wie Wikipedia!):**
    
    - **GND**: Gemeinsame Normdatei der Deutschen Nationalbibliothek
    - **VIAF**: Virtual International Authority File (OCLC)
    - **LOC**: Library of Congress Authority Files
    - **Getty TGN**: Thesaurus of Geographic Names
    - **Getty ULAN**: Union List of Artist Names
    
    Diese Quellen werden von Bibliothekaren und Fachleuten kuratiert
    und bieten höchste Datenqualität.
    """
    try:
        from src.datasources.authority import (
            AuthoritySourceManager,
            AuthoritySourceType,
        )
        
        # Map string to NodeType
        type_mapping = {
            "person": NodeType.PERSON,
            "event": NodeType.EVENT,
            "location": NodeType.LOCATION,
            "organization": NodeType.ORGANIZATION,
        }
        
        entity_type = type_mapping.get(request.entity_type.lower())
        if not entity_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid entity type: {request.entity_type}",
            )
        
        # Source mapping
        source_mapping = {
            "gnd": AuthoritySourceType.GND,
            "viaf": AuthoritySourceType.VIAF,
            "loc": AuthoritySourceType.LOC,
            "getty_tgn": AuthoritySourceType.GETTY_TGN,
            "getty_ulan": AuthoritySourceType.GETTY_ULAN,
        }
        
        manager = AuthoritySourceManager()
        manager.register_defaults()
        
        sources = [source_mapping[s] for s in request.sources if s in source_mapping]
        
        if not sources:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid sources specified",
            )
        
        # Search and import
        graph_manager: GraphManager = app.state.graph_manager
        result = await manager.import_to_database(
            query=request.query,
            entity_type=entity_type,
            graph_manager=graph_manager,
            sources=sources,
        )
        
        await manager.close_all()
        
        return {
            "success": result.get("success", False),
            "nodes_imported": result.get("nodes_imported", 0),
            "relationships_imported": result.get("relationships_imported", 0),
            "sources_used": result.get("sources_used", []),
            "authority_ids": result.get("authority_ids", []),
            "quality": "authoritative",  # Markiere als hochwertig
        }
        
    except ImportError as e:
        logger.error(f"Authority module not available: {e}")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authority sources module not available",
        )
    except Exception as e:
        logger.error(f"Authority import failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/sources/authority", tags=["Authority Sources"])
async def list_authority_sources():
    """
    Listet alle verfügbaren autoritativen Datenquellen.
    """
    return {
        "sources": [
            {
                "id": "gnd",
                "name": "GND - Gemeinsame Normdatei",
                "provider": "Deutsche Nationalbibliothek",
                "quality": "highest",
                "coverage": "German-speaking entities, universal",
                "url": "https://www.dnb.de/gnd",
            },
            {
                "id": "viaf",
                "name": "VIAF - Virtual International Authority File",
                "provider": "OCLC",
                "quality": "high",
                "coverage": "International, aggregated from national libraries",
                "url": "https://viaf.org",
            },
            {
                "id": "loc",
                "name": "LOC Authority Files",
                "provider": "Library of Congress",
                "quality": "highest",
                "coverage": "US standard, international coverage",
                "url": "https://id.loc.gov",
            },
            {
                "id": "getty_tgn",
                "name": "Getty TGN",
                "provider": "Getty Research Institute",
                "quality": "highest",
                "coverage": "Geographic names, historical places",
                "url": "http://vocab.getty.edu/tgn",
            },
            {
                "id": "getty_ulan",
                "name": "Getty ULAN",
                "provider": "Getty Research Institute",
                "quality": "highest",
                "coverage": "Artists, architects, art-related persons",
                "url": "http://vocab.getty.edu/ulan",
            },
        ],
        "note": "All sources are curated by librarians and domain experts - NOT crowdsourced",
    }


# =============================================================================
# Web UI
# =============================================================================

# Serve static files
static_path = Path(__file__).parent.parent / "web" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/ui", tags=["Web UI"])
async def serve_ui():
    """Serves the web-based user interface."""
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Web UI not found. Place index.html in src/web/static/",
    )


# =============================================================================
# Startup Command
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
