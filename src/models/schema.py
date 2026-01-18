"""
Pydantic Schema Definitionen für Knowledge Graph Entitäten und Beziehungen.

Diese Modelle werden für Structured Output vom LLM verwendet, um konsistente
und validierte Graphdaten zu erzeugen.
"""

from datetime import date
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS für strikte Typisierung
# =============================================================================


class NodeType(str, Enum):
    """Erlaubte Knotentypen im Knowledge Graph."""

    PERSON = "Person"
    EVENT = "Event"
    LOCATION = "Location"
    DATE = "Date"
    ORGANIZATION = "Organization"


class RelationType(str, Enum):
    """Erlaubte Beziehungstypen zwischen Knoten."""

    PARTICIPATED_IN = "PARTICIPATED_IN"
    HAPPENED_ON = "HAPPENED_ON"
    LOCATED_AT = "LOCATED_AT"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    # Erweiterte Beziehungen für historischen Kontext
    PRECEDED_BY = "PRECEDED_BY"
    FOLLOWED_BY = "FOLLOWED_BY"
    CAUSED = "CAUSED"
    BORN_IN = "BORN_IN"
    DIED_IN = "DIED_IN"


class SourceLabel(str, Enum):
    """Kennzeichnung der Datenquelle/Vertrauensstufe."""

    FACT = "Fact"  # Verifizierte Ground Truth
    CLAIM = "Claim"  # Zu überprüfende Behauptung


# =============================================================================
# BASE MODELS
# =============================================================================


class NodeBase(BaseModel):
    """Basisklasse für alle Knoten mit gemeinsamen Attributen."""

    id: UUID = Field(default_factory=uuid4, description="Eindeutige Knoten-ID")
    name: str = Field(
        ..., min_length=1, max_length=500, description="Primärer Name/Bezeichner"
    )
    description: str | None = Field(
        default=None, max_length=2000, description="Optionale Beschreibung"
    )
    source_label: SourceLabel = Field(
        default=SourceLabel.CLAIM, description="Fact oder Claim Kennzeichnung"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Konfidenz der Extraktion (0-1)"
    )
    aliases: list[str] = Field(
        default_factory=list, description="Alternative Namen/Schreibweisen"
    )


# =============================================================================
# NODE TYPES
# =============================================================================


class PersonNode(NodeBase):
    """Repräsentiert eine historische oder aktuelle Person."""

    node_type: Literal[NodeType.PERSON] = NodeType.PERSON
    birth_date: date | None = Field(
        default=None, description="Geburtsdatum (YYYY-MM-DD)"
    )
    death_date: date | None = Field(
        default=None, description="Sterbedatum (YYYY-MM-DD)"
    )
    nationality: str | None = Field(default=None, max_length=100)
    occupation: list[str] = Field(default_factory=list, description="Berufe/Rollen")


class EventNode(NodeBase):
    """Repräsentiert ein historisches Ereignis."""

    node_type: Literal[NodeType.EVENT] = NodeType.EVENT
    start_date: date | None = Field(default=None, description="Startdatum (YYYY-MM-DD)")
    end_date: date | None = Field(default=None, description="Enddatum (YYYY-MM-DD)")
    event_type: str | None = Field(
        default=None,
        description="Kategorie: war, treaty, election, discovery, etc.",
    )

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v: date | None, info) -> date | None:
        """Stellt sicher, dass end_date nach start_date liegt."""
        if v is not None and info.data.get("start_date") is not None:
            if v < info.data["start_date"]:
                raise ValueError("end_date must be after start_date")
        return v


class LocationNode(NodeBase):
    """Repräsentiert einen geografischen Ort."""

    node_type: Literal[NodeType.LOCATION] = NodeType.LOCATION
    location_type: str | None = Field(
        default=None,
        description="Typ: city, country, region, building, etc.",
    )
    coordinates: tuple[float, float] | None = Field(
        default=None, description="(latitude, longitude)"
    )
    parent_location: str | None = Field(
        default=None, description="Übergeordneter Ort (z.B. Land für Stadt)"
    )


class DateNode(NodeBase):
    """
    Repräsentiert ein spezifisches Datum oder Zeitraum.

    Wichtig für chronologische Konsistenzprüfungen.
    """

    node_type: Literal[NodeType.DATE] = NodeType.DATE
    date_value: date = Field(..., description="Exaktes Datum (YYYY-MM-DD)")
    precision: Literal["day", "month", "year", "decade", "century"] = Field(
        default="day", description="Genauigkeit der Datumsangabe"
    )
    calendar_system: str = Field(
        default="gregorian", description="Kalendersystem (gregorian, julian, etc.)"
    )


class OrganizationNode(NodeBase):
    """Repräsentiert eine Organisation, Institution oder Gruppe."""

    node_type: Literal[NodeType.ORGANIZATION] = NodeType.ORGANIZATION
    org_type: str | None = Field(
        default=None,
        description="Typ: government, military, company, political_party, etc.",
    )
    founded_date: date | None = Field(default=None)
    dissolved_date: date | None = Field(default=None)
    headquarters: str | None = Field(
        default=None, description="Hauptsitz (Location Name)"
    )


# Union Type für alle Nodes
AnyNode = Annotated[
    PersonNode | EventNode | LocationNode | DateNode | OrganizationNode,
    Field(discriminator="node_type"),
]


# =============================================================================
# RELATIONSHIP / EDGE MODEL
# =============================================================================


class Relationship(BaseModel):
    """
    Repräsentiert eine gerichtete Beziehung zwischen zwei Knoten.

    Das Triple-Format: (source) -[relationship]-> (target)
    """

    id: UUID = Field(default_factory=uuid4, description="Eindeutige Beziehungs-ID")
    source_name: str = Field(..., description="Name des Quellknotens")
    source_type: NodeType = Field(..., description="Typ des Quellknotens")
    relation_type: RelationType = Field(..., description="Art der Beziehung")
    target_name: str = Field(..., description="Name des Zielknotens")
    target_type: NodeType = Field(..., description="Typ des Zielknotens")

    # Metadaten
    source_label: SourceLabel = Field(default=SourceLabel.CLAIM)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Zusätzliche Edge-Properties"
    )

    @field_validator("source_name", "target_name")
    @classmethod
    def normalize_names(cls, v: str) -> str:
        """Normalisiert Namen für konsistente Suche."""
        return v.strip()


# =============================================================================
# EXTRACTION RESULT MODEL
# =============================================================================


class KnowledgeGraphExtraction(BaseModel):
    """
    Vollständiges Extraktionsergebnis vom LLM.

    Enthält alle extrahierten Knoten und Beziehungen aus einem Text.
    """

    nodes: list[AnyNode] = Field(
        default_factory=list, description="Extrahierte Entitäten"
    )
    relationships: list[Relationship] = Field(
        default_factory=list, description="Extrahierte Beziehungen"
    )
    source_text_hash: str | None = Field(
        default=None, description="Hash des Quelltexts für Deduplizierung"
    )
    extraction_metadata: dict[str, str | int | float] = Field(
        default_factory=dict, description="Metadaten zur Extraktion"
    )

    def get_triples(self) -> list[tuple[str, str, str]]:
        """Gibt Beziehungen als einfache (source, relation, target) Triples zurück."""
        return [
            (rel.source_name, rel.relation_type.value, rel.target_name)
            for rel in self.relationships
        ]

    def get_nodes_by_type(self, node_type: NodeType) -> list[AnyNode]:
        """Filtert Knoten nach Typ."""
        return [node for node in self.nodes if node.node_type == node_type]


# =============================================================================
# VERIFICATION MODELS (für spätere Erweiterung)
# =============================================================================


class VerificationResult(BaseModel):
    """Ergebnis der Verifikation einer Behauptung gegen Ground Truth."""

    claim_id: UUID
    status: Literal["verified", "contradicted", "unverifiable", "partially_verified"]
    confidence: float = Field(ge=0.0, le=1.0)
    matching_facts: list[UUID] = Field(default_factory=list)
    contradicting_facts: list[UUID] = Field(default_factory=list)
    explanation: str = Field(default="")


class ChronologicalCheck(BaseModel):
    """Ergebnis einer chronologischen Konsistenzprüfung."""

    is_consistent: bool
    issues: list[str] = Field(default_factory=list)
    timeline: list[tuple[date, str]] = Field(
        default_factory=list, description="Sortierte Zeitleiste der Ereignisse"
    )
