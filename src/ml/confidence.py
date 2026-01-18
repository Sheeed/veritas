"""
Machine Learning Modul für Konfidenzbewertung.

Implementiert:
- Feature Engineering für historische Claims
- Ensemble-Modell für Confidence Scoring
- Online Learning für kontinuierliche Verbesserung
- Explainability (SHAP-ähnliche Feature Importance)
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.models.schema import (
    AnyNode,
    EventNode,
    KnowledgeGraphExtraction,
    NodeType,
    PersonNode,
    Relationship,
)
from src.validation.validator import (
    FullValidationResult,
    ValidationIssue,
    IssueSeverity,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Engineering
# =============================================================================


class ClaimFeatures(BaseModel):
    """Extrahierte Features aus einem Claim für ML-Modelle."""

    # Strukturelle Features
    num_nodes: int = 0
    num_relationships: int = 0
    num_persons: int = 0
    num_events: int = 0
    num_locations: int = 0
    num_dates: int = 0
    num_organizations: int = 0

    # Vollständigkeits-Features
    nodes_with_dates: float = 0.0  # Anteil Nodes mit Datumsangaben
    relationships_per_node: float = 0.0
    avg_description_length: float = 0.0
    has_source_citations: bool = False

    # Zeitliche Features
    date_span_days: int | None = None  # Zeitspanne der erwähnten Daten
    dates_in_future: bool = False
    dates_very_old: bool = False  # Vor 500 v. Chr.

    # Sprachliche Features (vom Extraction Prompt)
    avg_entity_name_length: float = 0.0
    uses_vague_language: bool = False  # "vermutlich", "wahrscheinlich"
    has_specific_numbers: bool = False

    # Validierungs-Features (nach Validation)
    entity_match_rate: float = 0.0
    avg_entity_match_score: float = 0.0
    num_critical_issues: int = 0
    num_high_issues: int = 0
    num_chronological_issues: int = 0

    # Aggregierte Scores
    completeness_score: float = 0.0
    specificity_score: float = 0.0
    consistency_score: float = 0.0


class FeatureExtractor:
    """
    Extrahiert ML-Features aus Claims und Validierungsergebnissen.
    """

    VAGUE_TERMS = [
        "vermutlich",
        "wahrscheinlich",
        "möglicherweise",
        "etwa",
        "circa",
        "probably",
        "possibly",
        "around",
        "approximately",
        "supposedly",
        "allegedly",
        "perhaps",
        "might",
        "may have",
        "could have",
    ]

    def extract_from_extraction(
        self,
        extraction: KnowledgeGraphExtraction,
        source_text: str | None = None,
    ) -> ClaimFeatures:
        """Extrahiert Features aus einer KG-Extraktion."""
        features = ClaimFeatures()

        # Strukturelle Features
        features.num_nodes = len(extraction.nodes)
        features.num_relationships = len(extraction.relationships)

        features.num_persons = len(
            [n for n in extraction.nodes if n.node_type == NodeType.PERSON]
        )
        features.num_events = len(
            [n for n in extraction.nodes if n.node_type == NodeType.EVENT]
        )
        features.num_locations = len(
            [n for n in extraction.nodes if n.node_type == NodeType.LOCATION]
        )
        features.num_dates = len(
            [n for n in extraction.nodes if n.node_type == NodeType.DATE]
        )
        features.num_organizations = len(
            [n for n in extraction.nodes if n.node_type == NodeType.ORGANIZATION]
        )

        # Vollständigkeit
        if features.num_nodes > 0:
            features.relationships_per_node = (
                features.num_relationships / features.num_nodes
            )

            # Nodes mit Daten
            nodes_with_dates = 0
            for node in extraction.nodes:
                if isinstance(node, PersonNode) and (
                    node.birth_date or node.death_date
                ):
                    nodes_with_dates += 1
                elif isinstance(node, EventNode) and (node.start_date or node.end_date):
                    nodes_with_dates += 1
            features.nodes_with_dates = nodes_with_dates / features.num_nodes

            # Beschreibungslänge
            desc_lengths = [len(n.description or "") for n in extraction.nodes]
            features.avg_description_length = (
                sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0
            )

            # Entity-Namen
            name_lengths = [len(n.name) for n in extraction.nodes]
            features.avg_entity_name_length = sum(name_lengths) / len(name_lengths)

        # Zeitliche Features
        all_dates = self._collect_dates(extraction)
        if all_dates:
            features.date_span_days = (max(all_dates) - min(all_dates)).days
            features.dates_in_future = any(d > date.today() for d in all_dates)
            features.dates_very_old = any(d.year < -500 for d in all_dates)

        # Sprachliche Features (aus Quelltext)
        if source_text:
            text_lower = source_text.lower()
            features.uses_vague_language = any(
                term in text_lower for term in self.VAGUE_TERMS
            )
            features.has_specific_numbers = any(c.isdigit() for c in source_text)

        # Aggregierte Scores
        features.completeness_score = self._calc_completeness(features)
        features.specificity_score = self._calc_specificity(features)

        return features

    def enrich_with_validation(
        self,
        features: ClaimFeatures,
        validation_result: FullValidationResult,
    ) -> ClaimFeatures:
        """Reichert Features mit Validierungsergebnissen an."""
        # Entity Matching
        if validation_result.entity_matches:
            features.entity_match_rate = len(validation_result.entity_matches) / max(
                1, features.num_nodes
            )
            features.avg_entity_match_score = sum(
                m.match_score for m in validation_result.entity_matches
            ) / len(validation_result.entity_matches)

        # Issue Counts
        features.num_critical_issues = len(
            [
                i
                for i in validation_result.all_issues
                if i.severity == IssueSeverity.CRITICAL
            ]
        )
        features.num_high_issues = len(
            [
                i
                for i in validation_result.all_issues
                if i.severity == IssueSeverity.HIGH
            ]
        )
        features.num_chronological_issues = len(
            [
                i
                for i in validation_result.all_issues
                if "chronolog" in i.issue_type.value.lower()
            ]
        )

        # Consistency Score
        features.consistency_score = self._calc_consistency(features)

        return features

    def _collect_dates(self, extraction: KnowledgeGraphExtraction) -> list[date]:
        """Sammelt alle Daten aus einer Extraktion."""
        dates = []
        for node in extraction.nodes:
            if isinstance(node, PersonNode):
                if node.birth_date:
                    dates.append(node.birth_date)
                if node.death_date:
                    dates.append(node.death_date)
            elif isinstance(node, EventNode):
                if node.start_date:
                    dates.append(node.start_date)
                if node.end_date:
                    dates.append(node.end_date)
        return dates

    def _calc_completeness(self, features: ClaimFeatures) -> float:
        """Berechnet Vollständigkeits-Score."""
        factors = [
            min(1.0, features.num_nodes / 5),  # Mind. 5 Nodes
            features.nodes_with_dates,
            min(1.0, features.relationships_per_node),
            min(1.0, features.avg_description_length / 100),
        ]
        return sum(factors) / len(factors)

    def _calc_specificity(self, features: ClaimFeatures) -> float:
        """Berechnet Spezifitäts-Score."""
        score = 0.5  # Basis

        if features.has_specific_numbers:
            score += 0.2
        if not features.uses_vague_language:
            score += 0.2
        if features.num_dates > 0:
            score += 0.1

        return min(1.0, score)

    def _calc_consistency(self, features: ClaimFeatures) -> float:
        """Berechnet Konsistenz-Score basierend auf Validierung."""
        score = 1.0

        score -= features.num_critical_issues * 0.3
        score -= features.num_high_issues * 0.15
        score -= features.num_chronological_issues * 0.1

        if features.entity_match_rate > 0:
            score = score * 0.5 + features.avg_entity_match_score * 0.5

        return max(0.0, min(1.0, score))

    def to_vector(self, features: ClaimFeatures) -> np.ndarray:
        """Konvertiert Features in einen numerischen Vektor für ML."""
        return np.array(
            [
                features.num_nodes,
                features.num_relationships,
                features.num_persons,
                features.num_events,
                features.num_locations,
                features.num_dates,
                features.num_organizations,
                features.nodes_with_dates,
                features.relationships_per_node,
                features.avg_description_length / 100,  # Normalisiert
                1.0 if features.has_source_citations else 0.0,
                (
                    features.date_span_days / 365 if features.date_span_days else 0.0
                ),  # In Jahren
                1.0 if features.dates_in_future else 0.0,
                1.0 if features.dates_very_old else 0.0,
                features.avg_entity_name_length / 20,  # Normalisiert
                1.0 if features.uses_vague_language else 0.0,
                1.0 if features.has_specific_numbers else 0.0,
                features.entity_match_rate,
                features.avg_entity_match_score,
                features.num_critical_issues,
                features.num_high_issues,
                features.num_chronological_issues,
                features.completeness_score,
                features.specificity_score,
                features.consistency_score,
            ],
            dtype=np.float32,
        )


# =============================================================================
# Confidence Scoring Modelle
# =============================================================================


@dataclass
class TrainingExample:
    """Ein Trainingsbeispiel für das ML-Modell."""

    features: ClaimFeatures
    true_label: bool  # True = korrekt, False = falsch
    confidence: float  # 0-1 Konfidenz
    source: str = ""  # Quelle des Labels


class RuleBasedScorer:
    """
    Regelbasierter Scorer als Baseline und Fallback.
    """

    def score(self, features: ClaimFeatures) -> tuple[float, dict[str, float]]:
        """
        Berechnet Score basierend auf Regeln.

        Returns:
            (score, feature_contributions)
        """
        contributions = {}
        score = 0.5  # Basis

        # Kritische Issues sind stark negativ
        if features.num_critical_issues > 0:
            penalty = min(0.4, features.num_critical_issues * 0.2)
            score -= penalty
            contributions["critical_issues"] = -penalty

        # Hohe Issues
        if features.num_high_issues > 0:
            penalty = min(0.2, features.num_high_issues * 0.1)
            score -= penalty
            contributions["high_issues"] = -penalty

        # Entity Match Rate positiv
        if features.entity_match_rate > 0:
            bonus = features.entity_match_rate * 0.3
            score += bonus
            contributions["entity_match_rate"] = bonus

        # Vollständigkeit positiv
        bonus = features.completeness_score * 0.15
        score += bonus
        contributions["completeness"] = bonus

        # Spezifität positiv
        bonus = features.specificity_score * 0.1
        score += bonus
        contributions["specificity"] = bonus

        # Konsistenz
        contrib = (features.consistency_score - 0.5) * 0.2
        score += contrib
        contributions["consistency"] = contrib

        # Vage Sprache negativ
        if features.uses_vague_language:
            score -= 0.1
            contributions["vague_language"] = -0.1

        # Zukunftsdaten sehr negativ
        if features.dates_in_future:
            score -= 0.3
            contributions["future_dates"] = -0.3

        return max(0.0, min(1.0, score)), contributions


class EnsembleConfidenceScorer:
    """
    Ensemble-Modell für Konfidenz-Scoring.

    Kombiniert:
    1. Regelbasierter Scorer (interpretierbar)
    2. Logistische Regression (schnell, linear)
    3. Random Forest (nicht-linear)

    Mit Gewichtung basierend auf Trainingsdaten-Verfügbarkeit.
    """

    def __init__(self, model_path: str | None = None):
        self.rule_scorer = RuleBasedScorer()
        self.feature_extractor = FeatureExtractor()

        # ML-Modelle (optional trainiert)
        self._lr_weights: np.ndarray | None = None
        self._lr_bias: float = 0.0
        self._rf_trees: list[dict] | None = None

        # Ensemble-Gewichte
        self.weights = {
            "rules": 0.6,
            "lr": 0.2,
            "rf": 0.2,
        }

        # Training State
        self._training_examples: list[TrainingExample] = []
        self._is_trained = False

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def score(
        self,
        features: ClaimFeatures,
        explain: bool = False,
    ) -> dict[str, Any]:
        """
        Berechnet Konfidenz-Score für gegebene Features.

        Args:
            features: Extrahierte Claim-Features
            explain: Ob Erklärungen generiert werden sollen

        Returns:
            {
                "confidence": 0.0-1.0,
                "components": {...},
                "explanation": {...} (optional)
            }
        """
        feature_vector = self.feature_extractor.to_vector(features)

        # 1. Regelbasierter Score
        rule_score, rule_contributions = self.rule_scorer.score(features)

        # 2. Logistische Regression (falls trainiert)
        lr_score = rule_score  # Fallback
        if self._lr_weights is not None:
            z = np.dot(feature_vector, self._lr_weights) + self._lr_bias
            lr_score = 1 / (1 + np.exp(-z))

        # 3. Random Forest (falls trainiert)
        rf_score = rule_score  # Fallback
        if self._rf_trees:
            rf_score = self._predict_rf(feature_vector)

        # Ensemble kombinieren
        if self._is_trained:
            final_score = (
                self.weights["rules"] * rule_score
                + self.weights["lr"] * lr_score
                + self.weights["rf"] * rf_score
            )
        else:
            # Nur regelbasiert wenn nicht trainiert
            final_score = rule_score

        result = {
            "confidence": float(final_score),
            "components": {
                "rule_based": float(rule_score),
                "logistic_regression": float(lr_score),
                "random_forest": float(rf_score),
            },
            "is_trained": self._is_trained,
        }

        if explain:
            result["explanation"] = {
                "rule_contributions": rule_contributions,
                "top_features": self._get_top_features(features),
                "interpretation": self._generate_interpretation(final_score, features),
            }

        return result

    def score_extraction(
        self,
        extraction: KnowledgeGraphExtraction,
        source_text: str | None = None,
        validation_result: FullValidationResult | None = None,
    ) -> dict[str, Any]:
        """Convenience-Methode für direkte Extraktion-Bewertung."""
        features = self.feature_extractor.extract_from_extraction(
            extraction, source_text
        )

        if validation_result:
            features = self.feature_extractor.enrich_with_validation(
                features, validation_result
            )

        return self.score(features, explain=True)

    def add_training_example(
        self,
        features: ClaimFeatures,
        is_correct: bool,
        confidence: float = 1.0,
        source: str = "manual",
    ) -> None:
        """Fügt ein Trainingsbeispiel hinzu."""
        self._training_examples.append(
            TrainingExample(
                features=features,
                true_label=is_correct,
                confidence=confidence,
                source=source,
            )
        )

    def train(self, min_examples: int = 50) -> dict[str, Any]:
        """
        Trainiert die ML-Modelle auf den gesammelten Beispielen.

        Args:
            min_examples: Mindestanzahl Beispiele für Training

        Returns:
            Training-Statistiken
        """
        if len(self._training_examples) < min_examples:
            return {
                "success": False,
                "error": f"Need at least {min_examples} examples, have {len(self._training_examples)}",
            }

        # Daten vorbereiten
        X = np.array(
            [
                self.feature_extractor.to_vector(ex.features)
                for ex in self._training_examples
            ]
        )
        y = np.array(
            [ex.true_label for ex in self._training_examples], dtype=np.float32
        )
        weights = np.array([ex.confidence for ex in self._training_examples])

        # Logistische Regression trainieren (einfache Implementierung)
        self._train_lr(X, y, weights)

        # Random Forest trainieren (vereinfacht)
        self._train_rf(X, y)

        self._is_trained = True

        # Evaluation
        predictions = []
        for ex in self._training_examples:
            result = self.score(ex.features)
            predictions.append(result["confidence"] > 0.5)

        accuracy = sum(
            p == ex.true_label for p, ex in zip(predictions, self._training_examples)
        )
        accuracy /= len(self._training_examples)

        return {
            "success": True,
            "num_examples": len(self._training_examples),
            "training_accuracy": accuracy,
        }

    def _train_lr(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        """Trainiert Logistische Regression mit Gradient Descent."""
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
        """Trainiert einfachen Random Forest."""
        self._rf_trees = []
        n_samples = len(X)

        for _ in range(n_trees):
            # Bootstrap Sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Einfacher Decision Stump
            tree = self._build_stump(X_boot, y_boot)
            self._rf_trees.append(tree)

    def _build_stump(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Baut einen Decision Stump (einschichtiger Baum)."""
        best_feature = 0
        best_threshold = 0.5
        best_score = float("inf")

        # Zufällige Feature-Subset
        n_features = X.shape[1]
        feature_subset = np.random.choice(
            n_features, size=min(5, n_features), replace=False
        )

        for feature_idx in feature_subset:
            values = X[:, feature_idx]
            thresholds = np.percentile(values, [25, 50, 75])

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Gini Impurity
                left_p = np.mean(y[left_mask])
                right_p = np.mean(y[right_mask])

                gini_left = 2 * left_p * (1 - left_p)
                gini_right = 2 * right_p * (1 - right_p)

                score = (
                    np.sum(left_mask) * gini_left + np.sum(right_mask) * gini_right
                ) / len(y)

                if score < best_score:
                    best_score = score
                    best_feature = feature_idx
                    best_threshold = threshold

        left_mask = X[:, best_feature] <= best_threshold
        left_value = np.mean(y[left_mask]) if np.sum(left_mask) > 0 else 0.5
        right_value = np.mean(y[~left_mask]) if np.sum(~left_mask) > 0 else 0.5

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left_value": left_value,
            "right_value": right_value,
        }

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

        return np.mean(predictions)

    def _get_top_features(self, features: ClaimFeatures, top_n: int = 5) -> list[dict]:
        """Identifiziert die wichtigsten Features für die Bewertung."""
        feature_importance = []

        # Basierend auf Abweichung vom "neutralen" Wert
        if features.num_critical_issues > 0:
            feature_importance.append(
                {
                    "feature": "critical_issues",
                    "value": features.num_critical_issues,
                    "impact": "negative",
                    "importance": 0.9,
                }
            )

        if features.entity_match_rate > 0.5:
            feature_importance.append(
                {
                    "feature": "entity_match_rate",
                    "value": features.entity_match_rate,
                    "impact": "positive",
                    "importance": features.entity_match_rate,
                }
            )

        if features.dates_in_future:
            feature_importance.append(
                {
                    "feature": "dates_in_future",
                    "value": True,
                    "impact": "negative",
                    "importance": 0.8,
                }
            )

        if features.completeness_score > 0.7:
            feature_importance.append(
                {
                    "feature": "completeness_score",
                    "value": features.completeness_score,
                    "impact": "positive",
                    "importance": features.completeness_score * 0.5,
                }
            )

        # Sortieren und Top-N zurückgeben
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        return feature_importance[:top_n]

    def _generate_interpretation(self, score: float, features: ClaimFeatures) -> str:
        """Generiert menschenlesbare Interpretation."""
        if score >= 0.8:
            base = "High confidence: The claim appears well-supported"
        elif score >= 0.6:
            base = "Moderate confidence: Some aspects could be verified"
        elif score >= 0.4:
            base = "Low confidence: Significant uncertainties exist"
        else:
            base = "Very low confidence: Multiple issues detected"

        details = []
        if features.num_critical_issues > 0:
            details.append(f"{features.num_critical_issues} critical issues found")
        if features.entity_match_rate > 0.7:
            details.append(f"{features.entity_match_rate:.0%} of entities verified")
        if features.uses_vague_language:
            details.append("vague language detected")

        if details:
            base += f" ({', '.join(details)})"

        return base

    def save(self, path: str) -> None:
        """Speichert das Modell."""
        state = {
            "lr_weights": (
                self._lr_weights.tolist() if self._lr_weights is not None else None
            ),
            "lr_bias": self._lr_bias,
            "rf_trees": self._rf_trees,
            "weights": self.weights,
            "is_trained": self._is_trained,
            "n_examples": len(self._training_examples),
        }

        with open(path, "w") as f:
            json.dump(state, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Lädt das Modell."""
        with open(path, "r") as f:
            state = json.load(f)

        if state["lr_weights"]:
            self._lr_weights = np.array(state["lr_weights"], dtype=np.float32)
        self._lr_bias = state["lr_bias"]
        self._rf_trees = state["rf_trees"]
        self.weights = state["weights"]
        self._is_trained = state["is_trained"]

        logger.info(
            f"Model loaded from {path} (trained on {state['n_examples']} examples)"
        )
