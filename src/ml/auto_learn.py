"""
Veritas Auto-Learn System

Automatisches Lernen neuer Mythen aus Fact-Check Analysen.

Features:
- Extrahiert potenzielle neue Mythen aus LLM-Analysen
- Review-Queue für manuelle Überprüfung
- Automatisches Hinzufügen zur Datenbank nach Review
- Pattern Learning für bessere Erkennung
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.models.veritas_schema import (
    HistoricalMyth,
    MythOrigin,
    MythCategory,
    FactStatus,
    HistoricalEra,
    Region,
    Source,
    SourceType,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class LearnedMythCandidate(BaseModel):
    """Ein Kandidat für einen neuen Mythos."""

    id: str = Field(default_factory=lambda: str(uuid4())[:8])

    # Basis-Info
    claim: str
    claim_en: Optional[str] = None
    suggested_truth: str
    suggested_truth_en: Optional[str] = None

    # Klassifikation
    suggested_category: MythCategory = MythCategory.PERSON_MYTH
    suggested_era: HistoricalEra = HistoricalEra.MODERN
    suggested_status: FactStatus = FactStatus.MYTH

    # Meta
    source_text: str  # Original-Text der zur Erkennung führte
    extraction_date: datetime = Field(default_factory=datetime.now)
    confidence_score: float = 0.0

    # Keywords (automatisch extrahiert)
    extracted_keywords: List[str] = Field(default_factory=list)

    # Review Status
    review_status: str = "pending"  # pending, approved, rejected
    reviewer_notes: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class LearningStats(BaseModel):
    """Statistiken über das Learning System."""

    total_candidates: int = 0
    pending_review: int = 0
    approved: int = 0
    rejected: int = 0
    added_to_database: int = 0


# =============================================================================
# Auto-Learn System
# =============================================================================


class AutoLearnSystem:
    """
    System zum automatischen Lernen neuer Mythen.

    Workflow:
    1. Analyse identifiziert potentiellen Mythos
    2. System extrahiert relevante Infos
    3. Kandidat wird in Review-Queue gespeichert
    4. Nach Review: Zur Datenbank hinzufügen oder verwerfen
    """

    def __init__(self, data_dir: str = "data/learned"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.candidates_file = self.data_dir / "candidates.json"
        self.stats_file = self.data_dir / "stats.json"

        self._candidates: Dict[str, LearnedMythCandidate] = {}
        self._load_candidates()

    def _load_candidates(self) -> None:
        """Lädt gespeicherte Kandidaten."""
        if self.candidates_file.exists():
            try:
                with open(self.candidates_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        candidate = LearnedMythCandidate(**item)
                        self._candidates[candidate.id] = candidate
                logger.info(f"Loaded {len(self._candidates)} myth candidates")
            except Exception as e:
                logger.error(f"Failed to load candidates: {e}")

    def _save_candidates(self) -> None:
        """Speichert Kandidaten."""
        try:
            data = [c.model_dump() for c in self._candidates.values()]
            # Convert datetime to string
            for item in data:
                if item.get("extraction_date"):
                    item["extraction_date"] = (
                        item["extraction_date"].isoformat()
                        if isinstance(item["extraction_date"], datetime)
                        else item["extraction_date"]
                    )
                if item.get("reviewed_at"):
                    item["reviewed_at"] = (
                        item["reviewed_at"].isoformat()
                        if isinstance(item["reviewed_at"], datetime)
                        else item["reviewed_at"]
                    )

            with open(self.candidates_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Failed to save candidates: {e}")

    def extract_from_analysis(
        self,
        analysis_result: Dict[str, Any],
        source_text: str,
        min_confidence: float = 0.7,
    ) -> Optional[LearnedMythCandidate]:
        """
        Extrahiert einen Mythos-Kandidaten aus einer Analyse.

        Args:
            analysis_result: Ergebnis einer Veritas-Analyse
            source_text: Original-Text
            min_confidence: Mindest-Confidence für Kandidaten

        Returns:
            LearnedMythCandidate oder None
        """
        verdict = analysis_result.get("overall_verdict", "")

        # Nur bei klaren Ergebnissen lernen
        if verdict not in ["false", "myth", "propaganda"]:
            return None

        # Confidence prüfen
        claims = analysis_result.get("claims", [])
        if not claims:
            return None

        avg_confidence = sum(c.get("confidence_score", 0) for c in claims) / len(claims)
        if avg_confidence < min_confidence:
            return None

        # Hauptclaim extrahieren
        main_claim = claims[0] if claims else {}

        # Keywords extrahieren
        keywords = self._extract_keywords(source_text)

        # Kategorie erraten
        category = self._guess_category(source_text, main_claim)

        # Era erraten
        era = self._guess_era(source_text, main_claim)

        candidate = LearnedMythCandidate(
            claim=main_claim.get("original_claim", source_text[:200]),
            suggested_truth=analysis_result.get("summary_for_users", ""),
            suggested_category=category,
            suggested_era=era,
            suggested_status=FactStatus.MYTH if verdict == "myth" else FactStatus.FALSE,
            source_text=source_text,
            confidence_score=avg_confidence,
            extracted_keywords=keywords,
        )

        # Speichern
        self._candidates[candidate.id] = candidate
        self._save_candidates()

        logger.info(f"Extracted new myth candidate: {candidate.id}")
        return candidate

    def _extract_keywords(self, text: str) -> List[str]:
        """Extrahiert relevante Keywords aus Text."""
        import re

        # Einfache Keyword-Extraktion
        text_lower = text.lower()
        words = re.findall(r"\b[a-zA-ZäöüßÄÖÜ]{4,}\b", text_lower)

        # Stoppwörter entfernen
        stopwords = {
            "this",
            "that",
            "with",
            "from",
            "have",
            "been",
            "were",
            "they",
            "their",
            "what",
            "when",
            "where",
            "which",
            "would",
            "could",
            "should",
            "there",
            "these",
            "those",
            "being",
            "about",
            "dass",
            "eine",
            "einen",
            "einer",
            "einem",
            "sind",
            "wird",
            "wurde",
            "werden",
            "haben",
            "hatte",
            "sein",
            "seine",
            "seiner",
        }

        # Häufige historische Terme behalten
        historical_terms = {
            "war",
            "battle",
            "king",
            "emperor",
            "revolution",
            "century",
            "krieg",
            "schlacht",
            "könig",
            "kaiser",
            "revolution",
            "jahrhundert",
        }

        keywords = []
        word_counts = {}

        for word in words:
            if word in stopwords:
                continue
            word_counts[word] = word_counts.get(word, 0) + 1

        # Top Keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        for word, count in sorted_words[:10]:
            if count >= 1 or word in historical_terms:
                keywords.append(word)

        return keywords

    def _guess_category(self, text: str, claim: Dict) -> MythCategory:
        """Errät die Kategorie basierend auf Inhalt."""
        text_lower = text.lower()

        # Person indicators
        person_indicators = [
            "was",
            "war",
            "geboren",
            "born",
            "died",
            "starb",
            "said",
            "sagte",
        ]
        if any(ind in text_lower for ind in person_indicators):
            # Check if it's a quote
            if (
                '"' in text
                or "'" in text
                or "said" in text_lower
                or "sagte" in text_lower
            ):
                return MythCategory.QUOTE_MYTH
            return MythCategory.PERSON_MYTH

        # War indicators
        war_indicators = [
            "war",
            "battle",
            "krieg",
            "schlacht",
            "soldier",
            "soldat",
            "army",
            "armee",
        ]
        if any(ind in text_lower for ind in war_indicators):
            return MythCategory.WAR_MYTH

        # Event indicators
        event_indicators = [
            "happened",
            "occurred",
            "event",
            "ereignis",
            "discovery",
            "entdeckung",
        ]
        if any(ind in text_lower for ind in event_indicators):
            return MythCategory.EVENT_MYTH

        # Origin indicators
        origin_indicators = [
            "invented",
            "erfunden",
            "origin",
            "ursprung",
            "first",
            "erste",
        ]
        if any(ind in text_lower for ind in origin_indicators):
            return MythCategory.ORIGIN_MYTH

        return MythCategory.PERSON_MYTH  # Default

    def _guess_era(self, text: str, claim: Dict) -> HistoricalEra:
        """Errät die historische Ära."""
        text_lower = text.lower()

        # Ancient
        ancient = [
            "ancient",
            "antik",
            "roman",
            "römisch",
            "greek",
            "griechisch",
            "bc",
            "v.chr",
            "caesar",
            "cleopatra",
        ]
        if any(term in text_lower for term in ancient):
            return HistoricalEra.ANCIENT

        # Medieval
        medieval = [
            "medieval",
            "mittelalter",
            "knight",
            "ritter",
            "crusade",
            "kreuzzug",
            "castle",
            "burg",
        ]
        if any(term in text_lower for term in medieval):
            return HistoricalEra.MEDIEVAL

        # Early Modern
        early_modern = [
            "renaissance",
            "1500",
            "1600",
            "columbus",
            "kolumbus",
            "reformation",
        ]
        if any(term in text_lower for term in early_modern):
            return HistoricalEra.EARLY_MODERN

        # Modern
        modern = [
            "1700",
            "1800",
            "1900",
            "napoleon",
            "world war",
            "weltkrieg",
            "industrial",
            "industriell",
        ]
        if any(term in text_lower for term in modern):
            return HistoricalEra.MODERN

        # Contemporary
        contemporary = [
            "2000",
            "21st century",
            "21. jahrhundert",
            "internet",
            "digital",
        ]
        if any(term in text_lower for term in contemporary):
            return HistoricalEra.CONTEMPORARY

        return HistoricalEra.MODERN  # Default

    # =========================================================================
    # Review Operations
    # =========================================================================

    def get_pending_candidates(self) -> List[LearnedMythCandidate]:
        """Gibt alle ausstehenden Kandidaten zurück."""
        return [c for c in self._candidates.values() if c.review_status == "pending"]

    def get_candidate(self, candidate_id: str) -> Optional[LearnedMythCandidate]:
        """Gibt einen Kandidaten zurück."""
        return self._candidates.get(candidate_id)

    def approve_candidate(
        self,
        candidate_id: str,
        notes: Optional[str] = None,
        modified_claim: Optional[str] = None,
        modified_truth: Optional[str] = None,
    ) -> Optional[HistoricalMyth]:
        """
        Genehmigt einen Kandidaten und erstellt einen Mythos.

        Returns:
            Der erstellte HistoricalMyth oder None
        """
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            return None

        # Status aktualisieren
        candidate.review_status = "approved"
        candidate.reviewer_notes = notes
        candidate.reviewed_at = datetime.now()

        # Mythos erstellen
        myth = HistoricalMyth(
            id=f"learned_{candidate.id}",
            claim=modified_claim or candidate.claim,
            claim_en=candidate.claim_en,
            category=candidate.suggested_category,
            era=candidate.suggested_era,
            status=candidate.suggested_status,
            truth=modified_truth or candidate.suggested_truth,
            truth_en=candidate.suggested_truth_en,
            origin=MythOrigin(
                source="Veritas Auto-Learn",
                date=datetime.now().strftime("%Y-%m-%d"),
                reason="Automatically learned from fact-check analysis",
            ),
            sources=[
                Source(
                    type=SourceType.FACTCHECK,
                    title="Veritas Analysis",
                    reliability=ConfidenceLevel.MEDIUM,
                )
            ],
            keywords=candidate.extracted_keywords,
            popularity=50,  # Start mit mittlerer Popularität
        )

        self._save_candidates()
        logger.info(f"Approved candidate {candidate_id} as myth {myth.id}")

        return myth

    def reject_candidate(
        self,
        candidate_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Lehnt einen Kandidaten ab."""
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            return False

        candidate.review_status = "rejected"
        candidate.reviewer_notes = reason
        candidate.reviewed_at = datetime.now()

        self._save_candidates()
        logger.info(f"Rejected candidate {candidate_id}: {reason}")

        return True

    def get_stats(self) -> LearningStats:
        """Gibt Statistiken zurück."""
        candidates = list(self._candidates.values())

        return LearningStats(
            total_candidates=len(candidates),
            pending_review=len([c for c in candidates if c.review_status == "pending"]),
            approved=len([c for c in candidates if c.review_status == "approved"]),
            rejected=len([c for c in candidates if c.review_status == "rejected"]),
        )

    # =========================================================================
    # Integration with Database
    # =========================================================================

    def add_approved_to_database(self) -> Dict[str, int]:
        """
        Fügt alle genehmigten Kandidaten zur Mythen-Datenbank hinzu.

        Returns:
            Statistik über hinzugefügte Mythen
        """
        from src.data.myths_database import get_myths_database

        db = get_myths_database()
        added = 0

        for candidate in self._candidates.values():
            if candidate.review_status != "approved":
                continue

            myth_id = f"learned_{candidate.id}"

            # Prüfen ob schon existiert
            if myth_id in db.myths:
                continue

            # Mythos erstellen und hinzufügen
            myth = HistoricalMyth(
                id=myth_id,
                claim=candidate.claim,
                claim_en=candidate.claim_en,
                category=candidate.suggested_category,
                era=candidate.suggested_era,
                status=candidate.suggested_status,
                truth=candidate.suggested_truth,
                truth_en=candidate.suggested_truth_en,
                origin=MythOrigin(
                    source="Veritas Auto-Learn",
                    date=datetime.now().strftime("%Y-%m-%d"),
                    reason="Automatically learned from fact-check analysis",
                ),
                sources=[
                    Source(
                        type=SourceType.FACTCHECK,
                        title="Veritas Analysis",
                        reliability=ConfidenceLevel.MEDIUM,
                    )
                ],
                keywords=candidate.extracted_keywords,
                popularity=50,
            )

            db.myths[myth_id] = myth
            added += 1

        logger.info(f"Added {added} learned myths to database")
        return {"added": added}


# =============================================================================
# Singleton
# =============================================================================

_auto_learn_instance: Optional[AutoLearnSystem] = None


def get_auto_learn_system() -> AutoLearnSystem:
    """Gibt Auto-Learn System Instanz zurück."""
    global _auto_learn_instance
    if _auto_learn_instance is None:
        _auto_learn_instance = AutoLearnSystem()
    return _auto_learn_instance
