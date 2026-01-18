"""
Veritas Machine Learning Module

Enthält:
- VeritasConfidenceScorer: ML-basiertes Confidence Scoring
- PropagandaDetector: Propaganda-Pattern Erkennung
- AuthoritativeSourceVerifier: Verifikation gegen autoritative Quellen (KEIN Wikipedia!)
- AutoLearnSystem: Automatisches Mythen-Lernen
"""

from .veritas_confidence import (
    VeritasConfidenceScorer,
    VeritasFeatureExtractor,
    VeritasClaimFeatures,
    ConfidenceResult,
    get_confidence_scorer,
)

from .propaganda_detector import (
    PropagandaDetector,
    PropagandaTechnique,
    PropagandaAnalysis,
    PropagandaIndicator,
    get_propaganda_detector,
)

from .authoritative_verifier import (
    AuthoritativeSourceVerifier,
    AuthoritativeVerificationResult,
    AuthoritativeSourceResult,
    SourceTier,
    get_authoritative_verifier,
)

from .auto_learn import (
    AutoLearnSystem,
    LearnedMythCandidate,
    LearningStats,
    get_auto_learn_system,
)

__all__ = [
    # Confidence
    "VeritasConfidenceScorer",
    "VeritasFeatureExtractor", 
    "VeritasClaimFeatures",
    "ConfidenceResult",
    "get_confidence_scorer",
    
    # Propaganda
    "PropagandaDetector",
    "PropagandaTechnique",
    "PropagandaAnalysis",
    "PropagandaIndicator",
    "get_propaganda_detector",
    
    # Authoritative Verification (NO Wikipedia!)
    "AuthoritativeSourceVerifier",
    "AuthoritativeVerificationResult",
    "AuthoritativeSourceResult",
    "SourceTier",
    "get_authoritative_verifier",
    
    # Auto-Learn
    "AutoLearnSystem",
    "LearnedMythCandidate",
    "LearningStats",
    "get_auto_learn_system",
]