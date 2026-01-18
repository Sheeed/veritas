"""
Veritas - Erweiterte Datenmodelle

Strukturen fuer:
- Historische Mythen
- Narrative Patterns
- Mehrdimensionale Bewertungen
- Quellen-Tracking
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class VeritasBaseModel(BaseModel):
    """Basis-Model mit Enum-Serialisierung."""
    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# Enums fuer Bewertungssystem
# =============================================================================


class FactStatus(str, Enum):
    """Fakten-Status einer Behauptung."""
    CONFIRMED = "confirmed"           # Mehrere serioese Quellen stimmen ueberein
    LIKELY = "likely"                 # Gute Evidenz, aber nicht definitiv
    DISPUTED = "disputed"             # Historiker debattieren noch
    UNVERIFIED = "unverified"         # Keine Quellen gefunden
    FALSE = "false"                   # Widerlegt durch Quellen
    MYTH = "myth"                     # Populaerer Irrtum mit bekanntem Ursprung


class ContextStatus(str, Enum):
    """Kontext-Status einer Behauptung."""
    COMPLETE = "complete"             # Alle relevanten Aspekte erwaehnt
    SIMPLIFIED = "simplified"         # Korrekt aber oberflaechlich
    SELECTIVE = "selective"           # Wichtige Teile fehlen absichtlich
    DECONTEXTUALIZED = "decontextualized"  # Aus Zusammenhang gerissen
    MISLEADING = "misleading"         # Kontext absichtlich falsch dargestellt


class NarrativeStatus(str, Enum):
    """Narrativ-Status einer Behauptung."""
    NEUTRAL = "neutral"               # Keine erkennbare Agenda
    PERSPECTIVAL = "perspectival"     # Legitime Sichtweise, aber einseitig
    BIASED = "biased"                 # Deutliche Agenda erkennbar
    PROPAGANDA = "propaganda"         # Bekanntes Propagandamuster
    REVISIONISM = "revisionism"       # Versuch Geschichte umzuschreiben


class ConfidenceLevel(str, Enum):
    """Konfidenz-Level der Bewertung."""
    HIGH = "high"         # 90-100%: Mehrere Primaerquellen, Konsens
    MEDIUM = "medium"     # 60-89%: Gute Sekundaerquellen
    LOW = "low"           # 30-59%: Lueckenhafte Dokumentation
    UNCLEAR = "unclear"   # <30%: Nicht verifizierbar


class MythCategory(str, Enum):
    """Kategorien von historischen Mythen."""
    ORIGIN_MYTH = "origin_myth"           # Nationale Gruendungsmythen
    WAR_MYTH = "war_myth"                 # Kriegs-Mythen
    PERSON_MYTH = "person_myth"           # Personen-Mythen
    EVENT_MYTH = "event_myth"             # Ereignis-Mythen
    NUMBER_MYTH = "number_myth"           # Zahlen-Mythen
    QUOTE_MYTH = "quote_myth"             # Falsche Zitate
    CAUSATION_MYTH = "causation_myth"     # Falsche Kausalitaeten


class SourceType(str, Enum):
    """Typen von Quellen."""
    PRIMARY = "primary"           # Originalquelle, Dokument, Augenzeuge
    ACADEMIC = "academic"         # Peer-reviewed, wissenschaftlich
    ENCYCLOPEDIA = "encyclopedia" # Lexikon, Enzyklopaedie
    ARCHIVE = "archive"           # Archivmaterial
    FACTCHECK = "factcheck"       # Faktenchecker
    NEWS = "news"                 # Nachrichtenquelle
    AUTHORITY = "authority"       # Normdatei (GND, VIAF, etc.)


class HistoricalEra(str, Enum):
    """Historische Epochen."""
    ANCIENT = "ancient"               # Antike
    MEDIEVAL = "medieval"             # Mittelalter
    EARLY_MODERN = "early_modern"     # Fruehe Neuzeit
    MODERN = "modern"                 # 19./20. Jahrhundert
    CONTEMPORARY = "contemporary"     # Zeitgeschichte


class Region(str, Enum):
    """Geografische Regionen."""
    EUROPE = "europe"
    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"
    ASIA = "asia"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"
    OCEANIA = "oceania"
    GLOBAL = "global"


# =============================================================================
# Source Models
# =============================================================================


class Source(VeritasBaseModel):
    """Eine Quellenangabe."""
    
    type: SourceType
    title: str
    author: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    archive_id: Optional[str] = None
    reliability: ConfidenceLevel = ConfidenceLevel.MEDIUM
    quote: Optional[str] = None  # Relevantes Zitat
    page: Optional[str] = None


class SourceComparison(VeritasBaseModel):
    """Vergleich mehrerer Quellen zu einem Thema."""
    
    topic: str
    sources_analyzed: int
    consensus_points: list[str] = Field(default_factory=list)
    disagreements: list[dict[str, str]] = Field(default_factory=list)
    recommended_sources: list[Source] = Field(default_factory=list)


# =============================================================================
# Myth Models
# =============================================================================


class MythOrigin(VeritasBaseModel):
    """Ursprung eines Mythos."""
    
    source: str                       # Wo entstand der Mythos
    date: Optional[str] = None        # Wann (ca.)
    reason: str                       # Warum entstand er
    original_context: Optional[str] = None
    spread_mechanism: Optional[str] = None  # Wie verbreitete er sich


class HistoricalMyth(VeritasBaseModel):
    """Ein historischer Mythos oder Irrtum."""
    
    id: str                           # Eindeutige ID
    claim: str                        # Die Behauptung
    claim_en: Optional[str] = None    # Englische Version
    
    category: MythCategory
    era: HistoricalEra
    regions: list[Region] = Field(default_factory=list)
    
    status: FactStatus
    truth: str                        # Was stimmt wirklich
    truth_en: Optional[str] = None
    
    origin: MythOrigin
    
    sources: list[Source] = Field(default_factory=list)
    debunked_by: list[str] = Field(default_factory=list)  # Wer hat widerlegt
    
    related_myths: list[str] = Field(default_factory=list)  # IDs verwandter Mythen
    keywords: list[str] = Field(default_factory=list)
    
    legal_note: Optional[str] = None  # Z.B. bei Holocaust-Leugnung
    
    popularity: int = 0               # Wie verbreitet (0-100)
    last_seen: Optional[datetime] = None  # Wann zuletzt aufgetaucht
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "napoleon_height",
                "claim": "Napoleon war klein",
                "category": "person_myth",
                "era": "modern",
                "regions": ["europe"],
                "status": "myth",
                "truth": "Napoleon war 1,69m - ueberdurchschnittlich fuer seine Zeit",
                "origin": {
                    "source": "Britische Propaganda",
                    "date": "ca. 1803-1815",
                    "reason": "Kriegspropaganda"
                }
            }
        }


# =============================================================================
# Narrative Models
# =============================================================================


class NarrativePattern(VeritasBaseModel):
    """Ein bekanntes Narrativ-Muster."""
    
    id: str
    name: str
    name_en: Optional[str] = None
    description: str
    
    typical_claims: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    
    origin: Optional[str] = None
    purpose: Optional[str] = None     # Welchem Zweck dient das Narrativ
    
    examples: list[str] = Field(default_factory=list)  # Myth-IDs
    counter_narrative: Optional[str] = None


class NarrativeMatch(VeritasBaseModel):
    """Ergebnis einer Narrativ-Analyse."""
    
    matched_pattern: Optional[NarrativePattern] = None
    confidence: float = 0.0
    matching_elements: list[str] = Field(default_factory=list)
    
    is_known_propaganda: bool = False
    origin_tracking: Optional[str] = None


# =============================================================================
# Analysis Models
# =============================================================================


class ClaimAnalysis(VeritasBaseModel):
    """Detaillierte Analyse einer einzelnen Behauptung."""
    
    original_claim: str
    language: str = "de"
    
    # Extrahierte Elemente
    entities: list[dict[str, Any]] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    
    # Bewertungen
    fact_status: FactStatus
    context_status: ContextStatus
    narrative_status: NarrativeStatus
    confidence: ConfidenceLevel
    confidence_score: float = 0.0     # 0.0 - 1.0
    
    # Details
    what_is_true: list[str] = Field(default_factory=list)
    what_is_false: list[str] = Field(default_factory=list)
    what_is_missing: list[str] = Field(default_factory=list)
    
    # Narrativ
    narrative_match: Optional[NarrativeMatch] = None
    
    # Quellen
    sources_used: list[Source] = Field(default_factory=list)
    
    # Verwandte Mythen
    related_myths: list[str] = Field(default_factory=list)


class ContextAnalysis(VeritasBaseModel):
    """Analyse des fehlenden Kontexts."""
    
    missing_timeframe: Optional[str] = None
    missing_perspectives: list[str] = Field(default_factory=list)
    missing_causes: list[str] = Field(default_factory=list)
    missing_consequences: list[str] = Field(default_factory=list)
    
    selective_facts: list[str] = Field(default_factory=list)
    important_omissions: list[str] = Field(default_factory=list)


class FullAnalysis(VeritasBaseModel):
    """Vollstaendige Analyse einer Behauptung."""
    
    # Meta
    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    language: str = "de"
    
    # Input
    input_text: str
    input_type: str = "claim"  # claim, article, post
    
    # Gefundene Behauptungen
    claims_found: int = 1
    claims: list[ClaimAnalysis] = Field(default_factory=list)
    
    # Kontext
    context_analysis: Optional[ContextAnalysis] = None
    
    # Gesamtbewertung
    overall_verdict: str
    verdict_explanation: str
    
    # Fuer Laien
    summary_for_users: str
    
    # Quellen
    all_sources: list[Source] = Field(default_factory=list)
    
    # Empfehlung
    recommendation: Optional[str] = None


# =============================================================================
# Verdict Templates
# =============================================================================


class VerdictType(str, Enum):
    """Moegliche Gesamturteile."""
    
    HISTORICALLY_ACCURATE = "historically_accurate"
    SIMPLIFIED = "simplified"
    SELECTIVE = "selective"
    DECONTEXTUALIZED = "decontextualized"
    REINTERPRETED = "reinterpreted"
    PROPAGANDA = "propaganda"
    FALSE = "false"
    UNVERIFIABLE = "unverifiable"


VERDICT_DESCRIPTIONS = {
    VerdictType.HISTORICALLY_ACCURATE: {
        "de": "Historisch korrekt",
        "en": "Historically accurate",
        "description_de": "Fakten und Kontext stimmen mit dem historischen Konsens ueberein.",
        "description_en": "Facts and context align with historical consensus."
    },
    VerdictType.SIMPLIFIED: {
        "de": "Vereinfacht",
        "en": "Simplified",
        "description_de": "Grundsaetzlich korrekt, aber wichtige Nuancen fehlen.",
        "description_en": "Generally correct, but important nuances are missing."
    },
    VerdictType.SELECTIVE: {
        "de": "Selektiv",
        "en": "Selective",
        "description_de": "Nur ausgewaehlte Fakten werden praesentiert, wichtige Aspekte fehlen.",
        "description_en": "Only selected facts are presented, important aspects are missing."
    },
    VerdictType.DECONTEXTUALIZED: {
        "de": "Dekontextualisiert",
        "en": "Decontextualized",
        "description_de": "Fakten sind aus dem Zusammenhang gerissen.",
        "description_en": "Facts are taken out of context."
    },
    VerdictType.REINTERPRETED: {
        "de": "Umgedeutet",
        "en": "Reinterpreted",
        "description_de": "Fakten stimmen, aber die Interpretation ist fragwuerdig.",
        "description_en": "Facts are correct, but the interpretation is questionable."
    },
    VerdictType.PROPAGANDA: {
        "de": "Propaganda",
        "en": "Propaganda",
        "description_de": "Folgt bekannten manipulativen Narrativ-Mustern.",
        "description_en": "Follows known manipulative narrative patterns."
    },
    VerdictType.FALSE: {
        "de": "Falsch",
        "en": "False",
        "description_de": "Die Kernbehauptungen sind faktisch falsch.",
        "description_en": "The core claims are factually false."
    },
    VerdictType.UNVERIFIABLE: {
        "de": "Nicht verifizierbar",
        "en": "Unverifiable",
        "description_de": "Keine ausreichenden Quellen zur Pruefung vorhanden.",
        "description_en": "Insufficient sources available for verification."
    }
}


# =============================================================================
# Database Models
# =============================================================================


class MythDatabase(VeritasBaseModel):
    """Container fuer die Mythen-Datenbank."""
    
    version: str = "1.0.0"
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    myths: dict[str, HistoricalMyth] = Field(default_factory=dict)
    narratives: dict[str, NarrativePattern] = Field(default_factory=dict)
    
    def get_myth(self, myth_id: str) -> Optional[HistoricalMyth]:
        return self.myths.get(myth_id)
    
    def search_myths(self, query: str) -> list[HistoricalMyth]:
        """Sucht Mythen nach Keywords."""
        query_lower = query.lower()
        results = []
        for myth in self.myths.values():
            if (query_lower in myth.claim.lower() or
                query_lower in myth.truth.lower() or
                any(query_lower in kw.lower() for kw in myth.keywords)):
                results.append(myth)
        return results
    
    def get_myths_by_category(self, category: MythCategory) -> list[HistoricalMyth]:
        return [m for m in self.myths.values() if m.category == category]
    
    def get_myths_by_era(self, era: HistoricalEra) -> list[HistoricalMyth]:
        return [m for m in self.myths.values() if m.era == era]
