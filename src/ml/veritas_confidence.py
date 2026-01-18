"""
Veritas ML Confidence Scorer

Machine Learning basiertes Confidence Scoring für historische Faktenprüfung.

Features:
- Feature Engineering für historische Claims
- Ensemble-Modell (Logistische Regression + Random Forest)
- Explainability (Feature Importance)
- Online Learning für kontinuierliche Verbesserung
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List, Dict

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Models
# =============================================================================

class VeritasClaimFeatures(BaseModel):
    """Features für einen historischen Claim."""
    
    # Text Features
    text_length: int = 0
    word_count: int = 0
    sentence_count: int = 0
    has_numbers: bool = False
    has_dates: bool = False
    has_names: bool = False
    
    # Sprachliche Features
    uses_vague_language: bool = False
    uses_absolute_language: bool = False
    has_citations: bool = False
    question_marks: int = 0
    exclamation_marks: int = 0
    
    # Historische Features
    mentions_time_period: bool = False
    mentions_location: bool = False
    mentions_person: bool = False
    mentions_event: bool = False
    
    # Myth Database Match Features
    myth_match_score: float = 0.0
    keyword_match_count: int = 0
    is_known_myth: bool = False
    related_myths_count: int = 0
    
    # Source Features
    source_count: int = 0
    has_primary_source: bool = False
    has_academic_source: bool = False
    has_authority_source: bool = False
    
    # Validation Features
    entity_verification_score: float = 0.0
    date_consistency_score: float = 0.0
    narrative_pattern_match: float = 0.0
    
    # Aggregierte Scores
    specificity_score: float = 0.0
    credibility_score: float = 0.0
    completeness_score: float = 0.0


class ConfidenceResult(BaseModel):
    """Ergebnis der Confidence-Bewertung."""
    
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: str  # high, medium, low, unclear
    interpretation: str
    top_factors: List[Dict[str, Any]] = []
    features: Optional[VeritasClaimFeatures] = None
    

# =============================================================================
# Feature Extractor
# =============================================================================

class VeritasFeatureExtractor:
    """Extrahiert ML-Features aus Claims für Veritas."""
    
    VAGUE_TERMS = [
        "vermutlich", "wahrscheinlich", "möglicherweise", "etwa", "circa",
        "probably", "possibly", "around", "approximately", "supposedly",
        "allegedly", "perhaps", "might", "may have", "could have",
        "some say", "it is said", "legend has it", "reportedly",
        "angeblich", "soll", "man sagt", "der legende nach",
    ]
    
    ABSOLUTE_TERMS = [
        "always", "never", "every", "all", "none", "definitely",
        "certainly", "absolutely", "without doubt", "proven",
        "immer", "nie", "jeder", "alle", "keiner", "definitiv",
        "sicher", "bewiesen", "zweifellos",
    ]
    
    DATE_PATTERNS = [
        r'\d{1,2}\.\d{1,2}\.\d{2,4}',  # DD.MM.YYYY
        r'\d{4}',  # Jahr
        r'\d{1,2}(st|nd|rd|th)?\s+(century|jahrhundert)',
        r'(january|february|march|april|may|june|july|august|september|october|november|december)',
        r'(januar|februar|märz|april|mai|juni|juli|august|september|oktober|november|dezember)',
    ]
    
    NAME_PATTERNS = [
        r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # First Last
        r'(Kaiser|König|Präsident|General|Dr\.|Prof\.)\s+\w+',
        r'(Emperor|King|President|General|Dr\.|Prof\.)\s+\w+',
    ]
    
    def extract(
        self,
        text: str,
        myth_match: Optional[Dict] = None,
        sources: Optional[List[Dict]] = None,
        validation_result: Optional[Dict] = None,
    ) -> VeritasClaimFeatures:
        """Extrahiert alle Features aus einem Claim."""
        
        features = VeritasClaimFeatures()
        text_lower = text.lower()
        
        # === Text Features ===
        features.text_length = len(text)
        features.word_count = len(text.split())
        features.sentence_count = len(re.split(r'[.!?]+', text))
        features.has_numbers = bool(re.search(r'\d+', text))
        features.has_dates = any(re.search(p, text, re.IGNORECASE) for p in self.DATE_PATTERNS)
        features.has_names = bool(re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', text))
        
        # === Sprachliche Features ===
        features.uses_vague_language = any(term in text_lower for term in self.VAGUE_TERMS)
        features.uses_absolute_language = any(term in text_lower for term in self.ABSOLUTE_TERMS)
        features.has_citations = bool(re.search(r'\[\d+\]|\(\d{4}\)|according to|laut|nach', text_lower))
        features.question_marks = text.count('?')
        features.exclamation_marks = text.count('!')
        
        # === Historische Features ===
        time_indicators = ['century', 'jahrhundert', 'bc', 'ad', 'v.chr', 'n.chr', 'ancient', 'medieval', 'modern']
        features.mentions_time_period = any(ind in text_lower for ind in time_indicators)
        
        location_indicators = ['in', 'at', 'from', 'near', 'city', 'country', 'region']
        features.mentions_location = any(ind in text_lower for ind in location_indicators)
        
        features.mentions_person = bool(re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', text))
        
        event_indicators = ['war', 'battle', 'revolution', 'discovery', 'invention', 'death', 'birth']
        features.mentions_event = any(ind in text_lower for ind in event_indicators)
        
        # === Myth Match Features ===
        if myth_match:
            features.myth_match_score = myth_match.get('score', 0.0)
            features.keyword_match_count = myth_match.get('keyword_matches', 0)
            features.is_known_myth = myth_match.get('found', False)
            features.related_myths_count = len(myth_match.get('related_myths', []))
        
        # === Source Features ===
        if sources:
            features.source_count = len(sources)
            features.has_primary_source = any(s.get('type') == 'primary' for s in sources)
            features.has_academic_source = any(s.get('type') == 'academic' for s in sources)
            features.has_authority_source = any(s.get('type') == 'authority' for s in sources)
        
        # === Validation Features ===
        if validation_result:
            features.entity_verification_score = validation_result.get('entity_score', 0.0)
            features.date_consistency_score = validation_result.get('date_score', 0.0)
            features.narrative_pattern_match = validation_result.get('narrative_score', 0.0)
        
        # === Aggregierte Scores ===
        features.specificity_score = self._calc_specificity(features)
        features.credibility_score = self._calc_credibility(features)
        features.completeness_score = self._calc_completeness(features)
        
        return features
    
    def _calc_specificity(self, f: VeritasClaimFeatures) -> float:
        """Berechnet wie spezifisch der Claim ist."""
        score = 0.5
        
        if f.has_dates:
            score += 0.15
        if f.has_names:
            score += 0.15
        if f.has_numbers:
            score += 0.1
        if f.mentions_location:
            score += 0.1
        if f.uses_vague_language:
            score -= 0.2
        if f.uses_absolute_language:
            score -= 0.1  # Absolute Aussagen sind oft falsch
        
        return max(0.0, min(1.0, score))
    
    def _calc_credibility(self, f: VeritasClaimFeatures) -> float:
        """Berechnet Glaubwürdigkeit basierend auf Quellen und Verifizierung."""
        score = 0.3  # Basis ohne Quellen
        
        if f.has_primary_source:
            score += 0.25
        if f.has_academic_source:
            score += 0.2
        if f.has_authority_source:
            score += 0.15
        if f.has_citations:
            score += 0.1
        
        # Entity verification boost
        score += f.entity_verification_score * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calc_completeness(self, f: VeritasClaimFeatures) -> float:
        """Berechnet Vollständigkeit des Claims."""
        factors = [
            f.mentions_person,
            f.mentions_event,
            f.mentions_time_period,
            f.mentions_location,
            f.has_dates,
        ]
        
        return sum(factors) / len(factors)
    
    def to_vector(self, features: VeritasClaimFeatures) -> np.ndarray:
        """Konvertiert Features zu Numpy-Array für ML."""
        return np.array([
            features.text_length / 1000,  # Normalisiert
            features.word_count / 100,
            features.sentence_count / 10,
            float(features.has_numbers),
            float(features.has_dates),
            float(features.has_names),
            float(features.uses_vague_language),
            float(features.uses_absolute_language),
            float(features.has_citations),
            features.question_marks / 5,
            float(features.mentions_time_period),
            float(features.mentions_location),
            float(features.mentions_person),
            float(features.mentions_event),
            features.myth_match_score,
            features.keyword_match_count / 10,
            float(features.is_known_myth),
            features.related_myths_count / 5,
            features.source_count / 5,
            float(features.has_primary_source),
            float(features.has_academic_source),
            float(features.has_authority_source),
            features.entity_verification_score,
            features.date_consistency_score,
            features.narrative_pattern_match,
            features.specificity_score,
            features.credibility_score,
            features.completeness_score,
        ], dtype=np.float32)


# =============================================================================
# Training Example
# =============================================================================

@dataclass
class TrainingExample:
    """Ein Trainingsbeispiel für das ML-Modell."""
    features: VeritasClaimFeatures
    true_label: bool  # True = vertrauenswürdig, False = nicht vertrauenswürdig
    confidence: float = 1.0  # Gewichtung des Beispiels
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Ensemble Confidence Scorer
# =============================================================================

class VeritasConfidenceScorer:
    """
    Ensemble ML-Modell für Confidence Scoring.
    
    Kombiniert:
    - Regelbasierte Heuristiken
    - Logistische Regression
    - Random Forest
    
    Mit Explainability und Online Learning.
    """
    
    def __init__(self):
        self.feature_extractor = VeritasFeatureExtractor()
        
        # Modell-Gewichte
        self.weights = {
            "rules": 0.3,
            "lr": 0.35,
            "rf": 0.35,
        }
        
        # Logistische Regression
        self._lr_weights: Optional[np.ndarray] = None
        self._lr_bias: float = 0.0
        
        # Random Forest
        self._rf_trees: List[Dict] = []
        
        # Training State
        self._training_examples: List[TrainingExample] = []
        self._is_trained: bool = False
        
        # Cache
        self._cache: Dict[str, ConfidenceResult] = {}
    
    def score(
        self,
        text: str,
        myth_match: Optional[Dict] = None,
        sources: Optional[List[Dict]] = None,
        validation_result: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> ConfidenceResult:
        """
        Bewertet die Confidence eines Claims.
        
        Args:
            text: Der zu bewertende Claim
            myth_match: Optionales Ergebnis des Myth-Matchings
            sources: Optionale Quellen
            validation_result: Optionale Validierungsergebnisse
            use_cache: Cache nutzen
            
        Returns:
            ConfidenceResult mit Score und Erklärung
        """
        # Cache Check
        cache_key = hash(text)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Features extrahieren
        features = self.feature_extractor.extract(
            text=text,
            myth_match=myth_match,
            sources=sources,
            validation_result=validation_result,
        )
        
        # Scores von verschiedenen Methoden
        rule_score = self._rule_based_score(features)
        
        if self._is_trained:
            feature_vector = self.feature_extractor.to_vector(features)
            lr_score = self._predict_lr(feature_vector)
            rf_score = self._predict_rf(feature_vector)
        else:
            # Fallback wenn nicht trainiert
            lr_score = rule_score
            rf_score = rule_score
        
        # Ensemble
        final_score = (
            self.weights["rules"] * rule_score +
            self.weights["lr"] * lr_score +
            self.weights["rf"] * rf_score
        )
        
        # Confidence Level
        if final_score >= 0.8:
            level = "high"
        elif final_score >= 0.6:
            level = "medium"
        elif final_score >= 0.4:
            level = "low"
        else:
            level = "unclear"
        
        # Top Factors
        top_factors = self._get_top_factors(features, final_score)
        
        # Interpretation
        interpretation = self._generate_interpretation(final_score, features)
        
        result = ConfidenceResult(
            confidence=round(final_score, 3),
            confidence_level=level,
            interpretation=interpretation,
            top_factors=top_factors,
            features=features,
        )
        
        # Cache
        if use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def _rule_based_score(self, f: VeritasClaimFeatures) -> float:
        """Regelbasierter Score als Fallback und Ensemble-Teil."""
        score = 0.5
        
        # Positive Faktoren
        if f.is_known_myth:
            score -= 0.3  # Bekannter Mythos = niedriger Score für Wahrheit
        if f.has_primary_source:
            score += 0.15
        if f.has_academic_source:
            score += 0.1
        if f.entity_verification_score > 0.7:
            score += 0.15
        if f.specificity_score > 0.7:
            score += 0.1
        
        # Negative Faktoren
        if f.uses_vague_language:
            score -= 0.1
        if f.uses_absolute_language:
            score -= 0.05
        if f.narrative_pattern_match > 0.7:
            score -= 0.15  # Propaganda-Pattern
        if f.source_count == 0:
            score -= 0.15
        
        return max(0.0, min(1.0, score))
    
    def _predict_lr(self, x: np.ndarray) -> float:
        """Vorhersage mit Logistischer Regression."""
        if self._lr_weights is None:
            return 0.5
        
        z = np.dot(x, self._lr_weights) + self._lr_bias
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _predict_rf(self, x: np.ndarray) -> float:
        """Vorhersage mit Random Forest."""
        if not self._rf_trees:
            return 0.5
        
        predictions = []
        for tree in self._rf_trees:
            if x[tree["feature"]] <= tree["threshold"]:
                predictions.append(tree["left_value"])
            else:
                predictions.append(tree["right_value"])
        
        return float(np.mean(predictions))
    
    def _get_top_factors(self, f: VeritasClaimFeatures, score: float) -> List[Dict]:
        """Identifiziert die wichtigsten Faktoren."""
        factors = []
        
        if f.is_known_myth:
            factors.append({
                "factor": "known_myth",
                "impact": "negative",
                "importance": 0.9,
                "explanation": "This matches a known historical myth"
            })
        
        if f.has_primary_source:
            factors.append({
                "factor": "primary_source",
                "impact": "positive",
                "importance": 0.8,
                "explanation": "Supported by primary historical sources"
            })
        
        if f.has_academic_source:
            factors.append({
                "factor": "academic_source",
                "impact": "positive",
                "importance": 0.7,
                "explanation": "Referenced in academic literature"
            })
        
        if f.uses_vague_language:
            factors.append({
                "factor": "vague_language",
                "impact": "negative",
                "importance": 0.5,
                "explanation": "Uses vague or uncertain language"
            })
        
        if f.entity_verification_score > 0.7:
            factors.append({
                "factor": "verified_entities",
                "impact": "positive",
                "importance": f.entity_verification_score,
                "explanation": f"Entities verified ({f.entity_verification_score:.0%})"
            })
        
        if f.narrative_pattern_match > 0.5:
            factors.append({
                "factor": "propaganda_pattern",
                "impact": "negative",
                "importance": f.narrative_pattern_match,
                "explanation": "Matches known propaganda narrative pattern"
            })
        
        if f.specificity_score > 0.7:
            factors.append({
                "factor": "high_specificity",
                "impact": "positive",
                "importance": 0.6,
                "explanation": "Contains specific dates, names, and details"
            })
        
        # Sort by importance
        factors.sort(key=lambda x: x["importance"], reverse=True)
        return factors[:5]
    
    def _generate_interpretation(self, score: float, f: VeritasClaimFeatures) -> str:
        """Generiert menschenlesbare Interpretation."""
        if f.is_known_myth:
            return f"This claim matches a known historical myth. Confidence that it is TRUE: {score:.0%}"
        
        if score >= 0.8:
            base = "High confidence: This claim appears well-supported by historical evidence"
        elif score >= 0.6:
            base = "Moderate confidence: This claim has some supporting evidence but uncertainties remain"
        elif score >= 0.4:
            base = "Low confidence: This claim has significant uncertainties or contradictions"
        else:
            base = "Very low confidence: This claim appears unsupported or contradicts known facts"
        
        details = []
        if f.source_count > 0:
            details.append(f"{f.source_count} sources cited")
        if f.entity_verification_score > 0.5:
            details.append(f"entities {f.entity_verification_score:.0%} verified")
        if f.uses_vague_language:
            details.append("contains vague language")
        
        if details:
            base += f" ({', '.join(details)})"
        
        return base
    
    # =========================================================================
    # Online Learning
    # =========================================================================
    
    def add_training_example(
        self,
        text: str,
        is_true: bool,
        confidence: float = 1.0,
        myth_match: Optional[Dict] = None,
        sources: Optional[List[Dict]] = None,
    ) -> None:
        """Fügt ein Trainingsbeispiel hinzu."""
        features = self.feature_extractor.extract(
            text=text,
            myth_match=myth_match,
            sources=sources,
        )
        
        example = TrainingExample(
            features=features,
            true_label=is_true,
            confidence=confidence,
            metadata={"text": text[:100], "timestamp": datetime.now().isoformat()},
        )
        
        self._training_examples.append(example)
        logger.info(f"Added training example (total: {len(self._training_examples)})")
    
    def train(self, min_examples: int = 10) -> Dict[str, Any]:
        """Trainiert das Modell mit gesammelten Beispielen."""
        if len(self._training_examples) < min_examples:
            return {
                "success": False,
                "error": f"Need at least {min_examples} examples, have {len(self._training_examples)}",
            }
        
        # Feature-Matrix erstellen
        X = np.array([
            self.feature_extractor.to_vector(ex.features)
            for ex in self._training_examples
        ])
        y = np.array([float(ex.true_label) for ex in self._training_examples])
        weights = np.array([ex.confidence for ex in self._training_examples])
        
        # Logistische Regression trainieren
        self._train_lr(X, y, weights)
        
        # Random Forest trainieren
        self._train_rf(X, y)
        
        self._is_trained = True
        
        # Evaluation
        predictions = []
        for ex in self._training_examples:
            feature_vector = self.feature_extractor.to_vector(ex.features)
            pred = (
                self.weights["lr"] * self._predict_lr(feature_vector) +
                self.weights["rf"] * self._predict_rf(feature_vector)
            ) / (self.weights["lr"] + self.weights["rf"])
            predictions.append(pred > 0.5)
        
        accuracy = sum(
            p == ex.true_label 
            for p, ex in zip(predictions, self._training_examples)
        ) / len(self._training_examples)
        
        return {
            "success": True,
            "num_examples": len(self._training_examples),
            "training_accuracy": round(accuracy, 3),
        }
    
    def _train_lr(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        """Trainiert Logistische Regression."""
        n_features = X.shape[1]
        self._lr_weights = np.zeros(n_features, dtype=np.float32)
        self._lr_bias = 0.0
        
        learning_rate = 0.1
        n_iterations = 1000
        
        for _ in range(n_iterations):
            z = np.dot(X, self._lr_weights) + self._lr_bias
            predictions = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            
            errors = (predictions - y) * weights
            
            self._lr_weights -= learning_rate * np.dot(X.T, errors) / len(y)
            self._lr_bias -= learning_rate * np.mean(errors)
    
    def _train_rf(self, X: np.ndarray, y: np.ndarray, n_trees: int = 10) -> None:
        """Trainiert Random Forest."""
        self._rf_trees = []
        n_samples = len(X)
        
        for _ in range(n_trees):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            tree = self._build_stump(X_boot, y_boot)
            self._rf_trees.append(tree)
    
    def _build_stump(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Baut einen Decision Stump."""
        best_feature = 0
        best_threshold = 0.5
        best_score = float('inf')
        
        n_features = X.shape[1]
        feature_subset = np.random.choice(n_features, size=min(5, n_features), replace=False)
        
        for feature_idx in feature_subset:
            values = X[:, feature_idx]
            thresholds = np.percentile(values, [25, 50, 75])
            
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_p = np.mean(y[left_mask])
                right_p = np.mean(y[right_mask])
                
                gini_left = 2 * left_p * (1 - left_p)
                gini_right = 2 * right_p * (1 - right_p)
                
                score = (np.sum(left_mask) * gini_left + np.sum(right_mask) * gini_right) / len(y)
                
                if score < best_score:
                    best_score = score
                    best_feature = feature_idx
                    best_threshold = threshold
        
        left_mask = X[:, best_feature] <= best_threshold
        left_value = np.mean(y[left_mask]) if np.sum(left_mask) > 0 else 0.5
        right_value = np.mean(y[~left_mask]) if np.sum(~left_mask) > 0 else 0.5
        
        return {
            "feature": int(best_feature),
            "threshold": float(best_threshold),
            "left_value": float(left_value),
            "right_value": float(right_value),
        }
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save(self, path: str) -> None:
        """Speichert das Modell."""
        state = {
            "lr_weights": self._lr_weights.tolist() if self._lr_weights is not None else None,
            "lr_bias": self._lr_bias,
            "rf_trees": self._rf_trees,
            "weights": self.weights,
            "is_trained": self._is_trained,
            "n_examples": len(self._training_examples),
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> bool:
        """Lädt das Modell."""
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            if state.get("lr_weights"):
                self._lr_weights = np.array(state["lr_weights"], dtype=np.float32)
            self._lr_bias = state.get("lr_bias", 0.0)
            self._rf_trees = state.get("rf_trees", [])
            self.weights = state.get("weights", self.weights)
            self._is_trained = state.get("is_trained", False)
            
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Leert den Cache."""
        self._cache.clear()


# =============================================================================
# Singleton
# =============================================================================

_scorer_instance: Optional[VeritasConfidenceScorer] = None


def get_confidence_scorer() -> VeritasConfidenceScorer:
    """Gibt Confidence Scorer Instanz zurück."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = VeritasConfidenceScorer()
        # Versuche gespeichertes Modell zu laden
        model_path = Path("data/models/confidence_model.json")
        if model_path.exists():
            _scorer_instance.load(str(model_path))
    return _scorer_instance