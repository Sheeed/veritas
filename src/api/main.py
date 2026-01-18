"""
Veritas API v0.7.1 - Self-Improving Fact Checker

FastAPI Backend fuer Faktencheck mit:
- LLM (Groq)
- Wikidata SPARQL
- Wikipedia
- Local Knowledge Base
- Self-Learning
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown."""
    logger.info("=" * 50)
    logger.info("VERITAS API v0.7.1")
    logger.info("Self-Improving Fact Checker")
    logger.info("=" * 50)

    # Groq API Key
    if os.getenv("GROQ_API_KEY"):
        logger.info("[OK] Groq API key set")
        app.state.groq_available = True
    else:
        logger.warning("[!!] GROQ_API_KEY not set")
        app.state.groq_available = False

    logger.info("[OK] Wikidata SPARQL available")
    logger.info("[OK] Wikipedia API available")

    yield

    # Shutdown
    logger.info("Shutting down...")

    try:
        from src.ml.fact_checker import _fact_checker_instance

        if _fact_checker_instance:
            await _fact_checker_instance.close()
    except:
        pass

    try:
        from src.ml.self_improver import _instance

        if _instance:
            await _instance.close()
    except:
        pass


# =============================================================================
# App
# =============================================================================

app = FastAPI(
    title="Veritas API",
    description="""
# Veritas - Self-Improving Fact Checker

Faktencheck mit LLM + Wikidata + Self-Learning.

## Hauptendpunkte

| Endpoint | Beschreibung |
|----------|--------------|
| `POST /factcheck/check` | Vollstaendiger Faktencheck |
| `POST /factcheck/quick` | Schneller Check |
| `POST /learning/mine` | Fact Mining starten |
| `GET /learning/facts/stats` | KB Statistiken |

## Features

- Parallele Ausfuehrung (LLM + Wikidata + Wikipedia)
- Local Knowledge Base Lookup
- Offset-basiertes inkrementelles Mining
- Cache mit 24h TTL
    """,
    version="0.7.1",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Router
# =============================================================================

# Fact Check Router
try:
    from src.api.factcheck_router import router as factcheck_router

    app.include_router(factcheck_router)
    logger.info("[OK] /factcheck/* loaded")
except ImportError as e:
    logger.error(f"[!!] factcheck_router: {e}")

# Learning Router
try:
    from src.api.learning_router import router as learning_router

    app.include_router(learning_router)
    logger.info("[OK] /learning/* loaded")
except ImportError as e:
    logger.error(f"[!!] learning_router: {e}")

# Optional: Veritas Router (Mythen-DB)
try:
    from src.api.veritas_router import router as veritas_router

    app.include_router(veritas_router)
    logger.info("[OK] /veritas/* loaded")
except ImportError:
    pass

# Optional: Graph Router (Neo4j)
try:
    from src.api.graph_router import router as graph_router

    app.include_router(graph_router)
    logger.info("[OK] /graph/* loaded")
except ImportError:
    pass


# =============================================================================
# Core Endpoints
# =============================================================================


@app.get("/", tags=["Root"])
async def root():
    """API Info."""
    return {
        "name": "Veritas API",
        "version": "0.7.1",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "factcheck": "POST /factcheck/check",
            "learning": "POST /learning/mine",
            "stats": "GET /learning/facts/stats",
            "health": "GET /health",
        },
    }


@app.get("/health", tags=["Health"])
async def health():
    """System Health."""
    return {
        "status": "healthy",
        "version": "0.7.1",
        "groq_available": getattr(app.state, "groq_available", False),
        "wikidata_available": True,
    }


@app.get("/status", tags=["Health"])
async def detailed_status():
    """Detaillierter Status."""
    kb_stats = {}
    fc_stats = {}

    try:
        from src.ml.self_improver import get_self_improver

        kb_stats = get_self_improver().get_stats()
    except:
        pass

    try:
        from src.ml.fact_checker import get_fact_checker

        fc_stats = get_fact_checker().get_cache_stats()
    except:
        pass

    return {
        "version": "0.7.1",
        "knowledge_base": kb_stats,
        "fact_checker": fc_stats,
    }


# =============================================================================
# Error Handler
# =============================================================================


@app.exception_handler(Exception)
async def error_handler(request, exc):
    logger.error(f"Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": str(exc)},
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    print(
        f"""
    ========================================
      VERITAS API v0.7.1
      http://localhost:{port}
      http://localhost:{port}/docs
    ========================================
    """
    )

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=True)
