"""
Fortgeschrittene Validierungslogik für History Guardian.

Implementiert:
- Multi-Dimensionale Faktenprüfung
- Chronologische Konsistenzanalyse
- Entity Resolution / Fuzzy Matching
- Kontextuelle Plausibilitätsprüfung
- Widerspruchserkennung
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.db.graph_db import GraphManager
from src.models.schema import (
    AnyNode,
    DateNode,
    EventNode,
    KnowledgeGraphExtraction,
    LocationNode,
    NodeType,
    OrganizationNode,
    PersonNode,
    Relationship,
    RelationType,
    SourceLabel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums und Typen
# =============================================================================


class VerificationStatus(str, Enum):
    """Status einer Verifikation."""

    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIABLE = "unverifiable"
    SUSPICIOUS = "suspicious"


class IssueType(str, Enum):
    """Art eines erkannten Problems."""

    CHRONOLOGICAL_IMPOSSIBILITY = "chronological_impossibility"
    DATE_MISMATCH = "date_mismatch"
    LOCATION_MISMATCH = "location_mismatch"
    ENTITY_NOT_FOUND = "entity_not_found"
    RELATIONSHIP_CONTRADICTION = "relationship_contradiction"
    ANACHRONISM = "anachronism"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    FACTUAL_ERROR = "factual_error"


class IssueSeverity(str, Enum):
    """Schweregrad eines Problems."""

    CRITICAL = "critical"  # Definitiv falsch
    HIGH = "high"  # Wahrscheinlich falsch
    MEDIUM = "medium"  # Möglicherweise falsch
    LOW = "low"  # Kleine Unstimmigkeit
    INFO = "info"  # Hinweis, kein echtes Problem


# =============================================================================
# Datenmodelle
# =============================================================================


class ValidationIssue(BaseModel):
    """Ein erkanntes Validierungsproblem."""

    id: UUID = Field(default_factory=uuid4)
    issue_type: IssueType
    severity: IssueSeverity
    message: str
    claim_entity: str | None = None
    fact_entity: str | None = None
    claim_value: str | None = None
    fact_value: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    suggestion: str | None = None


class EntityMatch(BaseModel):
    """Ein Match zwischen Claim- und Fact-Entität."""

    claim_name: str
    claim_type: NodeType
    fact_name: str
    fact_type: NodeType
    match_score: float = Field(ge=0.0, le=1.0)
    match_method: str  # exact, alias, fuzzy, semantic
    fact_node_id: str | None = None


class ChronologyAnalysis(BaseModel):
    """Ergebnis der chronologischen Analyse."""

    is_consistent: bool
    timeline: list[tuple[str, str, str]]  # (date, entity, event)
    issues: list[ValidationIssue]
    earliest_date: date | None = None
    latest_date: date | None = None


class RelationshipValidation(BaseModel):
    """Validierung einer einzelnen Beziehung."""

    claim_relationship: dict[str, Any]
    status: VerificationStatus
    matching_facts: list[dict[str, Any]]
    contradicting_facts: list[dict[str, Any]]
    issues: list[ValidationIssue]
    confidence: float


class FullValidationResult(BaseModel):
    """Vollständiges Validierungsergebnis."""

    claim_id: str
    overall_status: VerificationStatus
    overall_confidence: float
    entity_matches: list[EntityMatch]
    relationship_validations: list[RelationshipValidation]
    chronology_analysis: ChronologyAnalysis
    all_issues: list[ValidationIssue]
    summary: str
    recommendation: str


# =============================================================================
# Entity Resolution
# =============================================================================


class EntityResolver:
    """
    Löst Entitäten zwischen Claims und Facts auf.

    Verwendet mehrere Matching-Strategien:
    1. Exaktes Matching (Name identisch)
    2. Alias-Matching (bekannte Aliase)
    3. Fuzzy-Matching (Levenshtein-Distanz)
    4. Phonetisches Matching (Soundex/Metaphone)
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalisiert Text für Vergleiche."""
        import re

        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _levenshtein_ratio(s1: str, s2: str) -> float:
        """Berechnet die Levenshtein-Ähnlichkeit (0-1)."""
        if not s1 or not s2:
            return 0.0

        len1, len2 = len(s1), len(s2)
        if len1 < len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1

        if len1 == 0:
            return 0.0

        distances = range(len2 + 1)
        for i, c1 in enumerate(s1):
            new_distances = [i + 1]
            for j, c2 in enumerate(s2):
                if c1 == c2:
                    new_distances.append(distances[j])
                else:
                    new_distances.append(
                        1 + min(distances[j], distances[j + 1], new_distances[-1])
                    )
            distances = new_distances

        return 1.0 - (distances[-1] / max(len1, len2))

    @staticmethod
    def _soundex(name: str) -> str:
        """Einfache Soundex-Implementierung für phonetisches Matching."""
        if not name:
            return ""

        name = name.upper()
        soundex = name[0]

        mapping = {
            "B": "1",
            "F": "1",
            "P": "1",
            "V": "1",
            "C": "2",
            "G": "2",
            "J": "2",
            "K": "2",
            "Q": "2",
            "S": "2",
            "X": "2",
            "Z": "2",
            "D": "3",
            "T": "3",
            "L": "4",
            "M": "5",
            "N": "5",
            "R": "6",
        }

        prev_code = mapping.get(name[0], "0")
        for char in name[1:]:
            code = mapping.get(char, "0")
            if code != "0" and code != prev_code:
                soundex += code
                if len(soundex) == 4:
                    break
            prev_code = code

        return soundex.ljust(4, "0")

    def find_matches(
        self,
        claim_name: str,
        claim_type: NodeType,
        fact_candidates: list[dict[str, Any]],
    ) -> list[EntityMatch]:
        """Findet passende Entitäten in den Fact-Kandidaten."""
        matches = []
        claim_normalized = self._normalize(claim_name)
        claim_soundex = self._soundex(claim_name)

        for fact in fact_candidates:
            fact_name = fact.get("name", "")
            fact_type_str = fact.get("node_type", "")
            fact_aliases = fact.get("aliases", [])

            # Type muss übereinstimmen
            try:
                fact_type = NodeType(fact_type_str) if fact_type_str else None
            except ValueError:
                continue

            if fact_type and fact_type != claim_type:
                continue

            fact_normalized = self._normalize(fact_name)
            best_score = 0.0
            best_method = ""

            # 1. Exaktes Matching
            if claim_normalized == fact_normalized:
                best_score = 1.0
                best_method = "exact"

            # 2. Alias-Matching
            if best_score < 1.0 and fact_aliases:
                for alias in fact_aliases:
                    if self._normalize(alias) == claim_normalized:
                        best_score = 0.95
                        best_method = "alias"
                        break

            # 3. Fuzzy-Matching
            if best_score < 0.95:
                fuzzy_score = self._levenshtein_ratio(claim_normalized, fact_normalized)
                if fuzzy_score > best_score:
                    best_score = fuzzy_score
                    best_method = "fuzzy"

            # 4. Phonetisches Matching (als Bonus)
            if best_score < self.similarity_threshold:
                if self._soundex(fact_name) == claim_soundex:
                    phonetic_score = 0.7
                    if phonetic_score > best_score:
                        best_score = phonetic_score
                        best_method = "phonetic"

            # Match hinzufügen wenn über Threshold
            if best_score >= self.similarity_threshold:
                matches.append(
                    EntityMatch(
                        claim_name=claim_name,
                        claim_type=claim_type,
                        fact_name=fact_name,
                        fact_type=fact_type or claim_type,
                        match_score=best_score,
                        match_method=best_method,
                        fact_node_id=fact.get("id"),
                    )
                )

        # Nach Score sortieren
        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches


# =============================================================================
# Chronologische Validierung
# =============================================================================


class ChronologyValidator:
    """
    Validiert chronologische Konsistenz von Behauptungen.

    Prüft:
    - Lebensdaten von Personen (Geburt vor Tod, Handlungen innerhalb Lebenszeit)
    - Event-Reihenfolgen (Ursache vor Wirkung)
    - Anachronismen (Technologien/Konzepte vor ihrer Erfindung)
    """

    # Bekannte historische Grenzdaten
    TECHNOLOGY_DATES: dict[str, date] = {
        "telephone": date(1876, 3, 10),
        "airplane": date(1903, 12, 17),
        "television": date(1927, 9, 7),
        "internet": date(1983, 1, 1),
        "smartphone": date(2007, 6, 29),
        "printing_press": date(1440, 1, 1),
        "steam_engine": date(1712, 1, 1),
        "electricity": date(1882, 9, 4),
        "automobile": date(1886, 1, 29),
        "computer": date(1945, 2, 15),
    }

    def __init__(self):
        self.issues: list[ValidationIssue] = []

    def _extract_dates(
        self, extraction: KnowledgeGraphExtraction
    ) -> dict[str, list[date]]:
        """Extrahiert alle Daten aus einer Extraktion, gruppiert nach Entität."""
        entity_dates: dict[str, list[date]] = {}

        for node in extraction.nodes:
            dates = []

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
            elif isinstance(node, DateNode):
                dates.append(node.date_value)
            elif isinstance(node, OrganizationNode):
                if node.founded_date:
                    dates.append(node.founded_date)
                if node.dissolved_date:
                    dates.append(node.dissolved_date)

            if dates:
                entity_dates[node.name] = dates

        return entity_dates

    def _check_person_lifespan(
        self,
        person: PersonNode,
        related_events: list[tuple[date, str]],
    ) -> list[ValidationIssue]:
        """Prüft ob Events innerhalb der Lebenszeit einer Person liegen."""
        issues = []

        # Geburt muss vor Tod liegen
        if person.birth_date and person.death_date:
            if person.birth_date > person.death_date:
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.CHRONOLOGICAL_IMPOSSIBILITY,
                        severity=IssueSeverity.CRITICAL,
                        message=f"Birth date ({person.birth_date}) is after death date ({person.death_date})",
                        claim_entity=person.name,
                        claim_value=str(person.birth_date),
                        fact_value=str(person.death_date),
                    )
                )

        # Events müssen innerhalb Lebenszeit liegen
        for event_date, event_name in related_events:
            if person.birth_date and event_date < person.birth_date:
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.CHRONOLOGICAL_IMPOSSIBILITY,
                        severity=IssueSeverity.CRITICAL,
                        message=f"{person.name} could not have participated in '{event_name}' ({event_date}) - born {person.birth_date}",
                        claim_entity=person.name,
                        claim_value=str(event_date),
                        fact_value=str(person.birth_date),
                        suggestion="Check birth date or event date",
                    )
                )

            if person.death_date and event_date > person.death_date:
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.CHRONOLOGICAL_IMPOSSIBILITY,
                        severity=IssueSeverity.CRITICAL,
                        message=f"{person.name} could not have participated in '{event_name}' ({event_date}) - died {person.death_date}",
                        claim_entity=person.name,
                        claim_value=str(event_date),
                        fact_value=str(person.death_date),
                        suggestion="Check death date or event date",
                    )
                )

        return issues

    def _check_anachronisms(
        self,
        extraction: KnowledgeGraphExtraction,
    ) -> list[ValidationIssue]:
        """Sucht nach Anachronismen (Technologien vor ihrer Erfindung)."""
        issues = []

        # Sammle alle erwähnten Daten
        earliest_date = None
        for node in extraction.nodes:
            node_date = None
            if isinstance(node, EventNode) and node.start_date:
                node_date = node.start_date
            elif isinstance(node, DateNode):
                node_date = node.date_value

            if node_date:
                if earliest_date is None or node_date < earliest_date:
                    earliest_date = node_date

        if not earliest_date:
            return issues

        # Prüfe Text auf Technologie-Erwähnungen (vereinfacht)
        all_text = " ".join([node.name.lower() for node in extraction.nodes])
        all_text += " " + " ".join(
            [(node.description or "").lower() for node in extraction.nodes]
        )

        for tech, invention_date in self.TECHNOLOGY_DATES.items():
            if tech in all_text and earliest_date < invention_date:
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.ANACHRONISM,
                        severity=IssueSeverity.HIGH,
                        message=f"'{tech}' mentioned in context of {earliest_date}, but invented {invention_date}",
                        claim_value=str(earliest_date),
                        fact_value=str(invention_date),
                        suggestion=f"'{tech}' was not available until {invention_date.year}",
                    )
                )

        return issues

    def validate(
        self,
        extraction: KnowledgeGraphExtraction,
        fact_extraction: KnowledgeGraphExtraction | None = None,
    ) -> ChronologyAnalysis:
        """
        Führt vollständige chronologische Validierung durch.
        """
        self.issues = []
        timeline: list[tuple[str, str, str]] = []

        # Timeline aufbauen
        all_dates: list[tuple[date, str, str]] = []

        for node in extraction.nodes:
            if isinstance(node, EventNode):
                if node.start_date:
                    all_dates.append((node.start_date, node.name, "start"))
                    timeline.append(
                        (node.start_date.isoformat(), node.name, "Event Start")
                    )
                if node.end_date:
                    all_dates.append((node.end_date, node.name, "end"))
            elif isinstance(node, PersonNode):
                if node.birth_date:
                    all_dates.append((node.birth_date, node.name, "birth"))
                    timeline.append((node.birth_date.isoformat(), node.name, "Birth"))
                if node.death_date:
                    all_dates.append((node.death_date, node.name, "death"))
                    timeline.append((node.death_date.isoformat(), node.name, "Death"))
            elif isinstance(node, DateNode):
                all_dates.append((node.date_value, node.name, "date"))

        # Timeline sortieren
        timeline.sort(key=lambda x: x[0])

        # Personen-Lebenszeitprüfung
        persons = [n for n in extraction.nodes if isinstance(n, PersonNode)]
        for person in persons:
            # Finde relevante Events für diese Person
            related_events = []
            for rel in extraction.relationships:
                if (
                    rel.source_name == person.name
                    and rel.relation_type == RelationType.PARTICIPATED_IN
                ):
                    # Finde Event-Datum
                    for d, name, dtype in all_dates:
                        if name == rel.target_name:
                            related_events.append((d, name))
                            break

            self.issues.extend(self._check_person_lifespan(person, related_events))

        # Anachronismus-Check
        self.issues.extend(self._check_anachronisms(extraction))

        # Wenn Fakten vorhanden, vergleiche Daten
        if fact_extraction:
            self._compare_with_facts(extraction, fact_extraction)

        # Earliest/Latest bestimmen
        dates_only = [d for d, _, _ in all_dates]

        return ChronologyAnalysis(
            is_consistent=len(
                [
                    i
                    for i in self.issues
                    if i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]
                ]
            )
            == 0,
            timeline=timeline,
            issues=self.issues,
            earliest_date=min(dates_only) if dates_only else None,
            latest_date=max(dates_only) if dates_only else None,
        )

    def _compare_with_facts(
        self,
        claim: KnowledgeGraphExtraction,
        fact: KnowledgeGraphExtraction,
    ) -> None:
        """Vergleicht Claim-Daten mit Fact-Daten."""
        # Baue Lookup für Fact-Daten
        fact_dates: dict[str, dict[str, date]] = {}

        for node in fact.nodes:
            dates_dict: dict[str, date] = {}
            if isinstance(node, PersonNode):
                if node.birth_date:
                    dates_dict["birth"] = node.birth_date
                if node.death_date:
                    dates_dict["death"] = node.death_date
            elif isinstance(node, EventNode):
                if node.start_date:
                    dates_dict["start"] = node.start_date
                if node.end_date:
                    dates_dict["end"] = node.end_date

            if dates_dict:
                fact_dates[node.name.lower()] = dates_dict

        # Vergleiche mit Claims
        for node in claim.nodes:
            key = node.name.lower()
            if key not in fact_dates:
                continue

            fact_date_dict = fact_dates[key]

            if isinstance(node, PersonNode):
                if node.birth_date and "birth" in fact_date_dict:
                    diff = abs((node.birth_date - fact_date_dict["birth"]).days)
                    if diff > 365:  # Mehr als 1 Jahr Unterschied
                        self.issues.append(
                            ValidationIssue(
                                issue_type=IssueType.DATE_MISMATCH,
                                severity=(
                                    IssueSeverity.HIGH
                                    if diff > 3650
                                    else IssueSeverity.MEDIUM
                                ),
                                message=f"Birth date mismatch for {node.name}",
                                claim_entity=node.name,
                                claim_value=str(node.birth_date),
                                fact_value=str(fact_date_dict["birth"]),
                                confidence=max(0.5, 1.0 - (diff / 3650)),
                            )
                        )


# =============================================================================
# Hauptvalidator
# =============================================================================


class ClaimValidator:
    """
    Hauptklasse für die vollständige Validierung von Claims.

    Kombiniert:
    - Entity Resolution
    - Chronologische Validierung
    - Beziehungsvalidierung
    - Plausibilitätsprüfung
    """

    def __init__(
        self,
        graph_manager: GraphManager,
        entity_resolver: EntityResolver | None = None,
        chronology_validator: ChronologyValidator | None = None,
    ):
        self.graph_manager = graph_manager
        self.entity_resolver = entity_resolver or EntityResolver()
        self.chronology_validator = chronology_validator or ChronologyValidator()

    async def validate_extraction(
        self,
        claim_extraction: KnowledgeGraphExtraction,
    ) -> FullValidationResult:
        """
        Validiert eine komplette Extraktion gegen die Ground Truth.
        """
        all_issues: list[ValidationIssue] = []
        entity_matches: list[EntityMatch] = []
        relationship_validations: list[RelationshipValidation] = []

        # 1. Entity Resolution
        for node in claim_extraction.nodes:
            candidates = await self.graph_manager.find_matching_facts(
                claim_name=node.name,
                claim_type=node.node_type,
            )

            matches = self.entity_resolver.find_matches(
                claim_name=node.name,
                claim_type=node.node_type,
                fact_candidates=candidates,
            )

            entity_matches.extend(matches)

            if not matches:
                all_issues.append(
                    ValidationIssue(
                        issue_type=IssueType.ENTITY_NOT_FOUND,
                        severity=IssueSeverity.MEDIUM,
                        message=f"No matching fact found for '{node.name}' ({node.node_type.value})",
                        claim_entity=node.name,
                    )
                )

        # 2. Chronologische Validierung
        chronology_result = self.chronology_validator.validate(claim_extraction)
        all_issues.extend(chronology_result.issues)

        # 3. Beziehungsvalidierung
        for rel in claim_extraction.relationships:
            rel_validation = await self._validate_relationship(rel, entity_matches)
            relationship_validations.append(rel_validation)
            all_issues.extend(rel_validation.issues)

        # 4. Gesamtstatus berechnen
        critical_issues = [
            i for i in all_issues if i.severity == IssueSeverity.CRITICAL
        ]
        high_issues = [i for i in all_issues if i.severity == IssueSeverity.HIGH]
        medium_issues = [i for i in all_issues if i.severity == IssueSeverity.MEDIUM]

        if critical_issues:
            overall_status = VerificationStatus.CONTRADICTED
            overall_confidence = 0.1
        elif high_issues:
            overall_status = VerificationStatus.SUSPICIOUS
            overall_confidence = 0.3
        elif medium_issues:
            overall_status = VerificationStatus.PARTIALLY_VERIFIED
            overall_confidence = 0.6
        elif entity_matches:
            overall_status = VerificationStatus.VERIFIED
            overall_confidence = min(
                1.0, sum(m.match_score for m in entity_matches) / len(entity_matches)
            )
        else:
            overall_status = VerificationStatus.UNVERIFIABLE
            overall_confidence = 0.5

        # 5. Summary und Recommendation
        summary = self._generate_summary(all_issues, entity_matches)
        recommendation = self._generate_recommendation(overall_status, all_issues)

        return FullValidationResult(
            claim_id=claim_extraction.source_text_hash or str(uuid4()),
            overall_status=overall_status,
            overall_confidence=overall_confidence,
            entity_matches=entity_matches,
            relationship_validations=relationship_validations,
            chronology_analysis=chronology_result,
            all_issues=all_issues,
            summary=summary,
            recommendation=recommendation,
        )

    async def _validate_relationship(
        self,
        rel: Relationship,
        entity_matches: list[EntityMatch],
    ) -> RelationshipValidation:
        """Validiert eine einzelne Beziehung."""
        issues = []
        matching_facts = []
        contradicting_facts = []

        # Finde gematchte Source und Target
        source_match = next(
            (m for m in entity_matches if m.claim_name == rel.source_name), None
        )
        target_match = next(
            (m for m in entity_matches if m.claim_name == rel.target_name), None
        )

        if not source_match or not target_match:
            return RelationshipValidation(
                claim_relationship=rel.model_dump(),
                status=VerificationStatus.UNVERIFIABLE,
                matching_facts=[],
                contradicting_facts=[],
                issues=[
                    ValidationIssue(
                        issue_type=IssueType.ENTITY_NOT_FOUND,
                        severity=IssueSeverity.MEDIUM,
                        message=f"Cannot verify relationship - entities not found in facts",
                    )
                ],
                confidence=0.5,
            )

        # Suche nach entsprechenden Fact-Beziehungen
        fact_rels = await self.graph_manager.get_node_relationships(
            node_name=source_match.fact_name,
            direction="out",
        )

        for fact_rel in fact_rels:
            if fact_rel.get("rel_type") == rel.relation_type.value:
                if (
                    fact_rel.get("other_name", "").lower()
                    == target_match.fact_name.lower()
                ):
                    matching_facts.append(fact_rel)

        # Status bestimmen
        if matching_facts:
            status = VerificationStatus.VERIFIED
            confidence = source_match.match_score * target_match.match_score
        elif contradicting_facts:
            status = VerificationStatus.CONTRADICTED
            confidence = 0.2
        else:
            status = VerificationStatus.UNVERIFIABLE
            confidence = 0.5

        return RelationshipValidation(
            claim_relationship=rel.model_dump(),
            status=status,
            matching_facts=matching_facts,
            contradicting_facts=contradicting_facts,
            issues=issues,
            confidence=confidence,
        )

    def _generate_summary(
        self,
        issues: list[ValidationIssue],
        matches: list[EntityMatch],
    ) -> str:
        """Generiert eine lesbare Zusammenfassung."""
        parts = []

        if matches:
            verified = [m for m in matches if m.match_score > 0.9]
            parts.append(
                f"{len(verified)}/{len(matches)} entities verified with high confidence"
            )

        critical = len([i for i in issues if i.severity == IssueSeverity.CRITICAL])
        if critical:
            parts.append(f"{critical} critical issues found")

        chronological = len(
            [i for i in issues if i.issue_type == IssueType.CHRONOLOGICAL_IMPOSSIBILITY]
        )
        if chronological:
            parts.append(f"{chronological} chronological impossibilities detected")

        return ". ".join(parts) if parts else "No significant issues found"

    def _generate_recommendation(
        self,
        status: VerificationStatus,
        issues: list[ValidationIssue],
    ) -> str:
        """Generiert eine Handlungsempfehlung."""
        if status == VerificationStatus.VERIFIED:
            return "The claim appears to be historically accurate."
        elif status == VerificationStatus.CONTRADICTED:
            return "The claim contains factual errors and should not be trusted."
        elif status == VerificationStatus.SUSPICIOUS:
            return "The claim contains potential inaccuracies. Manual verification recommended."
        elif status == VerificationStatus.PARTIALLY_VERIFIED:
            return "Some aspects of the claim could be verified, but others remain uncertain."
        else:
            return "Insufficient data to verify this claim. Additional sources needed."
