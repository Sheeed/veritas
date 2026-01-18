"""
Veritas Intelligence API Router

Endpoints für ML-basierte Features:
- Confidence Scoring
- Propaganda Detection
- Multi-Source Verification
- Auto-Learn System
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ConfidenceRequest(BaseModel):
    """Request für Confidence Scoring."""
    text: str = Field(min_length=10)
    include_features: bool = False


class ConfidenceResponse(BaseModel):
    """Response für Confidence Scoring."""
    confidence: float
    confidence_level: str
    interpretation: str
    top_factors: List[dict] = []
    features: Optional[dict] = None


class PropagandaRequest(BaseModel):
    """Request für Propaganda Detection."""
    text: str = Field(min_length=10)


class PropagandaResponse(BaseModel):
    """Response für Propaganda Detection."""
    is_propaganda: bool
    propaganda_score: float
    risk_level: str
    detected_techniques: List[dict] = []
    narrative_patterns: List[str] = []
    summary: str
    recommendation: str


class VerifyRequest(BaseModel):
    """Request für Authoritative Source Verification."""
    claim: str = Field(min_length=5)
    check_authority: bool = True   # Tier 1: GND, VIAF, LOC
    check_factcheck: bool = True   # Tier 2: Google Fact Check, ClaimBuster
    check_academic: bool = True    # Tier 3: CrossRef, Open Library
    skip_cache: bool = False


class VerifyResponse(BaseModel):
    """Response für Verification."""
    verified: bool
    confidence: float
    sources_checked: int
    sources_found: int
    highest_tier: Optional[str] = None
    fact_check_rating: Optional[str] = None
    summary: str
    recommendation: str
    results: List[dict] = []
    cached: bool = False


class LearnRequest(BaseModel):
    """Request für Auto-Learn."""
    text: str
    is_true: bool
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class CandidateReviewRequest(BaseModel):
    """Request für Candidate Review."""
    candidate_id: str
    action: str  # approve, reject
    notes: Optional[str] = None
    modified_claim: Optional[str] = None
    modified_truth: Optional[str] = None


# =============================================================================
# Confidence Scoring Endpoints
# =============================================================================

@router.post("/confidence", response_model=ConfidenceResponse)
async def score_confidence(request: ConfidenceRequest):
    """
    Bewertet die Confidence eines historischen Claims.
    
    Verwendet ein Ensemble aus:
    - Regelbasierte Heuristiken
    - Logistische Regression
    - Random Forest
    
    Returns:
        Confidence Score mit Erklärung und Top-Faktoren
    """
    try:
        from src.ml.veritas_confidence import get_confidence_scorer
        
        scorer = get_confidence_scorer()
        result = scorer.score(request.text)
        
        response_data = {
            "confidence": result.confidence,
            "confidence_level": result.confidence_level,
            "interpretation": result.interpretation,
            "top_factors": result.top_factors,
        }
        
        if request.include_features and result.features:
            response_data["features"] = result.features.model_dump()
        
        return ConfidenceResponse(**response_data)
        
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Confidence scorer not available"
        )
    except Exception as e:
        logger.error(f"Confidence scoring failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/confidence/train")
async def train_confidence_model(min_examples: int = 10):
    """
    Trainiert das Confidence-Modell mit gesammelten Beispielen.
    
    Benötigt mindestens 10 Trainingsbeispiele.
    """
    try:
        from src.ml.veritas_confidence import get_confidence_scorer
        
        scorer = get_confidence_scorer()
        result = scorer.train(min_examples=min_examples)
        
        if result["success"]:
            # Modell speichern
            scorer.save("data/models/confidence_model.json")
        
        return result
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# Propaganda Detection Endpoints
# =============================================================================

@router.post("/propaganda", response_model=PropagandaResponse)
async def detect_propaganda(request: PropagandaRequest):
    """
    Analysiert einen Text auf Propaganda-Muster.
    
    Erkennt:
    - Manipulative Sprache
    - Logische Fehlschlüsse
    - Bekannte Propaganda-Narrative
    - Emotionale Manipulation
    """
    try:
        from src.ml.propaganda_detector import get_propaganda_detector
        
        detector = get_propaganda_detector()
        result = detector.analyze(request.text)
        
        return PropagandaResponse(
            is_propaganda=result.is_propaganda,
            propaganda_score=result.propaganda_score,
            risk_level=result.risk_level,
            detected_techniques=[t.model_dump() for t in result.detected_techniques],
            narrative_patterns=result.narrative_patterns,
            summary=result.summary,
            recommendation=result.recommendation,
        )
        
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Propaganda detector not available"
        )
    except Exception as e:
        logger.error(f"Propaganda detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/propaganda/techniques")
async def list_propaganda_techniques():
    """
    Listet alle erkennbaren Propaganda-Techniken.
    """
    try:
        from src.ml.propaganda_detector import get_propaganda_detector, PropagandaTechnique
        
        detector = get_propaganda_detector()
        
        techniques = []
        for technique in PropagandaTechnique:
            info = detector.get_technique_info(technique)
            techniques.append({
                "id": technique.value,
                "name": info["name"],
                "description": info["description"],
            })
        
        return {"techniques": techniques}
        
    except Exception as e:
        logger.error(f"Failed to list techniques: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# Multi-Source Verification Endpoints
# =============================================================================

@router.post("/verify", response_model=VerifyResponse)
async def verify_claim(request: VerifyRequest):
    """
    Verifiziert einen Claim gegen AUTORITATIVE Quellen.
    
    KEINE Wikipedia/Wikidata - nur vertrauenswürdige Quellen!
    
    Quellen-Hierarchie:
    - **Tier 1 (95%)**: GND, VIAF, Library of Congress
    - **Tier 2 (85%)**: Google Fact Check, ClaimBuster
    - **Tier 3 (80%)**: CrossRef (Academic), Open Library
    
    Results werden gecached (24h TTL).
    """
    try:
        from src.ml.authoritative_verifier import get_authoritative_verifier
        
        verifier = get_authoritative_verifier()
        result = await verifier.verify(
            claim=request.claim,
            check_authority=request.check_authority,
            check_factcheck=request.check_factcheck,
            check_academic=request.check_academic,
            skip_cache=request.skip_cache,
        )
        
        return VerifyResponse(
            verified=result.verified,
            confidence=result.confidence,
            sources_checked=result.sources_checked,
            sources_found=result.sources_found,
            highest_tier=result.highest_tier_found.value if result.highest_tier_found else None,
            fact_check_rating=result.fact_check_rating,
            summary=result.summary,
            recommendation=result.recommendation,
            results=[r.model_dump() for r in result.results],
            cached=result.cached,
        )
        
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authoritative verifier not available"
        )
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/verify/cache")
async def clear_verification_cache():
    """
    Leert den Verifikations-Cache.
    """
    try:
        from src.ml.authoritative_verifier import get_authoritative_verifier
        
        verifier = get_authoritative_verifier()
        if verifier.cache:
            count = verifier.cache.clear()
            return {"cleared": count}
        
        return {"cleared": 0}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/verify/tiers")
async def get_source_tiers():
    """
    Gibt Informationen über die Quellen-Hierarchie zurück.
    """
    return {
        "tiers": [
            {
                "tier": "tier_1_authority",
                "name": "Authority Files",
                "reliability": 0.95,
                "sources": ["GND (Deutsche Nationalbibliothek)", "VIAF", "Library of Congress"],
                "description": "Authoritative catalogs maintained by national libraries",
            },
            {
                "tier": "tier_2_factcheck",
                "name": "Fact-Check Organizations",
                "reliability": 0.85,
                "sources": ["Google Fact Check API (aggregates Snopes, PolitiFact, etc.)", "ClaimBuster"],
                "description": "Professional fact-checking organizations",
            },
            {
                "tier": "tier_3_academic",
                "name": "Academic Sources",
                "reliability": 0.80,
                "sources": ["CrossRef (DOI registry)", "Open Library"],
                "description": "Peer-reviewed academic publications and scholarly books",
            },
        ],
        "note": "Wikipedia and Wikidata are NOT used as they are not authoritative primary sources."
    }


# =============================================================================
# Auto-Learn Endpoints
# =============================================================================

@router.post("/learn")
async def add_learning_example(request: LearnRequest):
    """
    Fügt ein Trainingsbeispiel zum Auto-Learn System hinzu.
    
    Diese Beispiele werden für das ML-Training verwendet.
    """
    try:
        from src.ml.veritas_confidence import get_confidence_scorer
        
        scorer = get_confidence_scorer()
        scorer.add_training_example(
            text=request.text,
            is_true=request.is_true,
            confidence=request.confidence,
        )
        
        return {"success": True, "message": "Training example added"}
        
    except Exception as e:
        logger.error(f"Failed to add learning example: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/learn/candidates")
async def get_learning_candidates():
    """
    Gibt alle ausstehenden Mythos-Kandidaten zum Review zurück.
    """
    try:
        from src.ml.auto_learn import get_auto_learn_system
        
        system = get_auto_learn_system()
        candidates = system.get_pending_candidates()
        
        return {
            "count": len(candidates),
            "candidates": [c.model_dump() for c in candidates],
        }
        
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auto-learn system not available"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/learn/review")
async def review_candidate(request: CandidateReviewRequest):
    """
    Reviewed einen Mythos-Kandidaten (approve/reject).
    """
    try:
        from src.ml.auto_learn import get_auto_learn_system
        
        system = get_auto_learn_system()
        
        if request.action == "approve":
            myth = system.approve_candidate(
                candidate_id=request.candidate_id,
                notes=request.notes,
                modified_claim=request.modified_claim,
                modified_truth=request.modified_truth,
            )
            
            if myth:
                return {
                    "success": True,
                    "action": "approved",
                    "myth_id": myth.id,
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Candidate not found"
                )
                
        elif request.action == "reject":
            success = system.reject_candidate(
                candidate_id=request.candidate_id,
                reason=request.notes,
            )
            
            return {
                "success": success,
                "action": "rejected",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Action must be 'approve' or 'reject'"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/learn/sync-database")
async def sync_learned_to_database():
    """
    Synchronisiert genehmigte Kandidaten zur Mythen-Datenbank.
    """
    try:
        from src.ml.auto_learn import get_auto_learn_system
        
        system = get_auto_learn_system()
        result = system.add_approved_to_database()
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/learn/stats")
async def get_learning_stats():
    """
    Gibt Statistiken über das Learning System zurück.
    """
    try:
        from src.ml.auto_learn import get_auto_learn_system
        
        system = get_auto_learn_system()
        stats = system.get_stats()
        
        return stats.model_dump()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# Combined Analysis
# =============================================================================

@router.post("/analyze")
async def full_intelligence_analysis(
    text: str,
    include_verification: bool = True,
    include_propaganda: bool = True,
):
    """
    Führt eine vollständige Intelligence-Analyse durch.
    
    Kombiniert:
    - Confidence Scoring
    - Propaganda Detection
    - Multi-Source Verification (optional)
    
    Returns:
        Kombiniertes Analyse-Ergebnis
    """
    results = {"text": text[:200]}
    
    # Confidence
    try:
        from src.ml.veritas_confidence import get_confidence_scorer
        scorer = get_confidence_scorer()
        confidence_result = scorer.score(text)
        results["confidence"] = {
            "score": confidence_result.confidence,
            "level": confidence_result.confidence_level,
            "interpretation": confidence_result.interpretation,
        }
    except Exception as e:
        results["confidence"] = {"error": str(e)}
    
    # Propaganda
    if include_propaganda:
        try:
            from src.ml.propaganda_detector import get_propaganda_detector
            detector = get_propaganda_detector()
            propaganda_result = detector.analyze(text)
            results["propaganda"] = {
                "is_propaganda": propaganda_result.is_propaganda,
                "score": propaganda_result.propaganda_score,
                "risk_level": propaganda_result.risk_level,
                "summary": propaganda_result.summary,
            }
        except Exception as e:
            results["propaganda"] = {"error": str(e)}
    
    # Verification (Authoritative Sources Only!)
    if include_verification:
        try:
            from src.ml.authoritative_verifier import get_authoritative_verifier
            verifier = get_authoritative_verifier()
            verify_result = await verifier.verify(text)
            results["verification"] = {
                "verified": verify_result.verified,
                "confidence": verify_result.confidence,
                "sources_found": verify_result.sources_found,
                "highest_tier": verify_result.highest_tier_found.value if verify_result.highest_tier_found else None,
                "fact_check_rating": verify_result.fact_check_rating,
                "summary": verify_result.summary,
            }
        except Exception as e:
            results["verification"] = {"error": str(e)}
    
    # Overall Assessment
    try:
        confidence_score = results.get("confidence", {}).get("score", 0.5)
        propaganda_score = results.get("propaganda", {}).get("score", 0)
        verify_confidence = results.get("verification", {}).get("confidence", 0.5)
        
        # Gewichteter Gesamtscore
        overall = (confidence_score * 0.3 + (1 - propaganda_score) * 0.3 + verify_confidence * 0.4)
        
        if overall >= 0.7:
            assessment = "HIGH CREDIBILITY"
        elif overall >= 0.5:
            assessment = "MODERATE CREDIBILITY"
        elif overall >= 0.3:
            assessment = "LOW CREDIBILITY"
        else:
            assessment = "VERY LOW CREDIBILITY"
        
        results["overall"] = {
            "score": round(overall, 3),
            "assessment": assessment,
        }
    except:
        pass
    
    return results