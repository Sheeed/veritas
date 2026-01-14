"""
Autoritative Datenquellen für History Guardian.

Diese Quellen sind kuratiert von Bibliothekaren und Fachleuten,
NICHT crowdsourced wie Wikipedia/Wikidata.

Unterstützte Quellen:
- GND (Gemeinsame Normdatei) - Deutsche Nationalbibliothek
- VIAF (Virtual International Authority File) - OCLC
- LOC (Library of Congress) Authority Files
- Getty Vocabularies (TGN, ULAN, AAT)
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any
from xml.etree import ElementTree as ET

import httpx
from pydantic import BaseModel, Field

from src.models.schema import (
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


class AuthoritySourceType(str, Enum):
    """Autoritative Datenquellen (NICHT crowdsourced)."""
    GND = "gnd"  # Deutsche Nationalbibliothek
    VIAF = "viaf"  # Virtual International Authority File
    LOC = "loc"  # Library of Congress
    GETTY_TGN = "getty_tgn"  # Getty Thesaurus of Geographic Names
    GETTY_ULAN = "getty_ulan"  # Union List of Artist Names
    GETTY_AAT = "getty_aat"  # Art & Architecture Thesaurus


class QualityLevel(str, Enum):
    """Qualitätsstufe der Daten."""
    VERIFIED = "verified"  # Von Fachleuten geprüft
    AUTHORITATIVE = "authoritative"  # Aus Autoritätsdatei
    DERIVED = "derived"  # Abgeleitet aus mehreren Quellen
    UNCERTAIN = "uncertain"  # Unsichere Zuordnung


@dataclass
class AuthorityRecord:
    """Ein Datensatz aus einer Autoritätsdatei."""
    source: AuthoritySourceType
    authority_id: str  # z.B. GND-ID, VIAF-ID, LCCN
    name: str
    name_variants: list[str]
    entity_type: NodeType
    birth_date: date | None = None
    death_date: date | None = None
    description: str | None = None
    quality: QualityLevel = QualityLevel.AUTHORITATIVE
    raw_data: dict[str, Any] | None = None
    
    @property
    def source_url(self) -> str:
        """Generiert die URL zur Originalquelle."""
        urls = {
            AuthoritySourceType.GND: f"https://d-nb.info/gnd/{self.authority_id}",
            AuthoritySourceType.VIAF: f"https://viaf.org/viaf/{self.authority_id}",
            AuthoritySourceType.LOC: f"https://id.loc.gov/authorities/names/{self.authority_id}",
            AuthoritySourceType.GETTY_TGN: f"http://vocab.getty.edu/tgn/{self.authority_id}",
            AuthoritySourceType.GETTY_ULAN: f"http://vocab.getty.edu/ulan/{self.authority_id}",
        }
        return urls.get(self.source, "")


class AuthorityImportResult(BaseModel):
    """Ergebnis eines Authority-Imports."""
    source: AuthoritySourceType
    query: str
    success: bool
    records_found: int = 0
    nodes_created: int = 0
    extraction: KnowledgeGraphExtraction | None = None
    authority_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# =============================================================================
# Basis-Klasse
# =============================================================================


class AuthoritySource(ABC):
    """Abstrakte Basisklasse für Autoritätsdatenquellen."""
    
    source_type: AuthoritySourceType
    base_url: str
    
    def __init__(self):
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "HistoryGuardian/1.0 (Academic Research Tool)",
                    "Accept": "application/json, application/xml",
                },
            )
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    @abstractmethod
    async def search(self, query: str, entity_type: NodeType) -> list[AuthorityRecord]:
        """Sucht nach Entitäten."""
        pass
    
    @abstractmethod
    async def get_by_id(self, authority_id: str) -> AuthorityRecord | None:
        """Holt einen Datensatz nach ID."""
        pass
    
    def record_to_node(self, record: AuthorityRecord) -> PersonNode | EventNode | LocationNode | OrganizationNode | None:
        """Konvertiert einen Authority Record in einen Node."""
        
        # Gemeinsame Attribute
        base_attrs = {
            "name": record.name,
            "aliases": record.name_variants[:5] if record.name_variants else [],
            "description": record.description,
            "source_label": SourceLabel.FACT,  # Authority = verifiziert
            "confidence": 1.0,  # Höchste Konfidenz für Autoritätsdaten
        }
        
        if record.entity_type == NodeType.PERSON:
            return PersonNode(
                **base_attrs,
                birth_date=record.birth_date,
                death_date=record.death_date,
            )
        
        elif record.entity_type == NodeType.LOCATION:
            return LocationNode(**base_attrs)
        
        elif record.entity_type == NodeType.ORGANIZATION:
            return OrganizationNode(**base_attrs)
        
        elif record.entity_type == NodeType.EVENT:
            return EventNode(
                **base_attrs,
                start_date=record.birth_date,  # Reuse für Event-Datum
                end_date=record.death_date,
            )
        
        return None
    
    async def import_to_graph(
        self,
        query: str,
        entity_type: NodeType,
    ) -> AuthorityImportResult:
        """Importiert Suchergebnisse als Knowledge Graph."""
        errors = []
        nodes = []
        authority_ids = []
        
        try:
            records = await self.search(query, entity_type)
            
            for record in records:
                try:
                    node = self.record_to_node(record)
                    if node:
                        nodes.append(node)
                        authority_ids.append(record.authority_id)
                except Exception as e:
                    errors.append(f"Record conversion failed: {e}")
            
            extraction = KnowledgeGraphExtraction(
                nodes=nodes,
                relationships=[],
                extraction_metadata={
                    "source": self.source_type.value,
                    "query": query,
                    "quality": "authoritative",
                },
            )
            
            return AuthorityImportResult(
                source=self.source_type,
                query=query,
                success=True,
                records_found=len(records),
                nodes_created=len(nodes),
                extraction=extraction,
                authority_ids=authority_ids,
                errors=errors,
            )
            
        except Exception as e:
            logger.error(f"{self.source_type} import failed: {e}")
            return AuthorityImportResult(
                source=self.source_type,
                query=query,
                success=False,
                errors=[str(e)],
            )


# =============================================================================
# GND - Gemeinsame Normdatei (Deutsche Nationalbibliothek)
# =============================================================================


class GNDSource(AuthoritySource):
    """
    GND - Gemeinsame Normdatei der Deutschen Nationalbibliothek.
    
    Höchste Qualitätsstufe für deutschsprachige Entitäten.
    Verwendet den lobid-gnd API Service.
    """
    
    source_type = AuthoritySourceType.GND
    base_url = "https://lobid.org/gnd"
    
    # GND Entity Types
    GND_TYPES = {
        NodeType.PERSON: "Person",
        NodeType.LOCATION: "PlaceOrGeographicName",
        NodeType.ORGANIZATION: "CorporateBody",
        NodeType.EVENT: "SubjectHeading",
    }
    
    async def search(self, query: str, entity_type: NodeType) -> list[AuthorityRecord]:
        """Sucht in der GND."""
        client = await self._get_client()
        
        gnd_type = self.GND_TYPES.get(entity_type, "")
        
        params = {
            "q": query,
            "filter": f"type:{gnd_type}" if gnd_type else None,
            "size": 10,
            "format": "json",
        }
        params = {k: v for k, v in params.items() if v}
        
        response = await client.get(f"{self.base_url}/search", params=params)
        response.raise_for_status()
        
        data = response.json()
        records = []
        
        for item in data.get("member", []):
            record = self._parse_gnd_record(item, entity_type)
            if record:
                records.append(record)
        
        return records
    
    async def get_by_id(self, gnd_id: str) -> AuthorityRecord | None:
        """Holt einen GND-Datensatz nach ID."""
        client = await self._get_client()
        
        response = await client.get(f"{self.base_url}/{gnd_id}.json")
        
        if response.status_code == 404:
            return None
        
        response.raise_for_status()
        data = response.json()
        
        # Typ bestimmen
        types = data.get("type", [])
        if "Person" in types:
            entity_type = NodeType.PERSON
        elif "PlaceOrGeographicName" in types:
            entity_type = NodeType.LOCATION
        elif "CorporateBody" in types:
            entity_type = NodeType.ORGANIZATION
        else:
            entity_type = NodeType.PERSON
        
        return self._parse_gnd_record(data, entity_type)
    
    def _parse_gnd_record(self, data: dict, entity_type: NodeType) -> AuthorityRecord | None:
        """Parst einen GND-Datensatz."""
        gnd_id = data.get("gndIdentifier")
        if not gnd_id:
            return None
        
        # Name
        name = data.get("preferredName", "")
        if not name:
            name = data.get("preferredNameForThePerson", "")
        
        # Varianten
        variants = []
        for variant in data.get("variantName", []):
            if isinstance(variant, str):
                variants.append(variant)
        
        # Daten
        birth_date = self._parse_gnd_date(data.get("dateOfBirth", []))
        death_date = self._parse_gnd_date(data.get("dateOfDeath", []))
        
        # Beschreibung
        description_parts = []
        if data.get("professionOrOccupation"):
            occupations = [o.get("label", "") for o in data.get("professionOrOccupation", []) if isinstance(o, dict)]
            if occupations:
                description_parts.append(", ".join(occupations[:3]))
        if data.get("biographicalOrHistoricalInformation"):
            description_parts.append(data["biographicalOrHistoricalInformation"][:200])
        
        return AuthorityRecord(
            source=AuthoritySourceType.GND,
            authority_id=gnd_id,
            name=name,
            name_variants=variants,
            entity_type=entity_type,
            birth_date=birth_date,
            death_date=death_date,
            description=" | ".join(description_parts) if description_parts else None,
            quality=QualityLevel.VERIFIED,
            raw_data=data,
        )
    
    def _parse_gnd_date(self, date_values: list) -> date | None:
        """Parst ein GND-Datum."""
        if not date_values:
            return None
        
        date_str = date_values[0] if isinstance(date_values, list) else date_values
        
        if not isinstance(date_str, str):
            return None
        
        # Verschiedene Formate
        patterns = [
            (r"^(\d{4})-(\d{2})-(\d{2})$", lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
            (r"^(\d{4})-(\d{2})$", lambda m: date(int(m[1]), int(m[2]), 1)),
            (r"^(\d{4})$", lambda m: date(int(m[1]), 1, 1)),
            (r"^-(\d{4})$", lambda m: date(-int(m[1]), 1, 1)),  # v. Chr.
        ]
        
        for pattern, converter in patterns:
            match = re.match(pattern, date_str)
            if match:
                try:
                    return converter(match)
                except ValueError:
                    continue
        
        return None


# =============================================================================
# VIAF - Virtual International Authority File
# =============================================================================


class VIAFSource(AuthoritySource):
    """
    VIAF - Virtual International Authority File.
    
    Aggregiert Normdaten aus nationalen Bibliotheken weltweit.
    Sehr hohe Qualität durch institutionelle Zusammenarbeit.
    """
    
    source_type = AuthoritySourceType.VIAF
    base_url = "https://viaf.org/viaf"
    
    async def search(self, query: str, entity_type: NodeType) -> list[AuthorityRecord]:
        """Sucht in VIAF."""
        client = await self._get_client()
        
        # VIAF Search API
        search_url = "https://viaf.org/viaf/search"
        
        # CQL Query
        cql_index = {
            NodeType.PERSON: "local.personalNames",
            NodeType.ORGANIZATION: "local.corporateNames",
            NodeType.LOCATION: "local.geographicNames",
        }.get(entity_type, "cql.any")
        
        params = {
            "query": f'{cql_index} all "{query}"',
            "maximumRecords": 10,
            "httpAccept": "application/json",
        }
        
        response = await client.get(search_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        records = []
        
        search_results = data.get("searchRetrieveResponse", {}).get("records", [])
        
        for item in search_results:
            record_data = item.get("record", {}).get("recordData", {})
            record = self._parse_viaf_record(record_data, entity_type)
            if record:
                records.append(record)
        
        return records
    
    async def get_by_id(self, viaf_id: str) -> AuthorityRecord | None:
        """Holt einen VIAF-Datensatz nach ID."""
        client = await self._get_client()
        
        response = await client.get(
            f"{self.base_url}/{viaf_id}/viaf.json",
            follow_redirects=True,
        )
        
        if response.status_code == 404:
            return None
        
        response.raise_for_status()
        data = response.json()
        
        return self._parse_viaf_record(data, NodeType.PERSON)
    
    def _parse_viaf_record(self, data: dict, entity_type: NodeType) -> AuthorityRecord | None:
        """Parst einen VIAF-Datensatz."""
        viaf_id = data.get("viafID")
        if not viaf_id:
            return None
        
        # Name aus mainHeadings
        name = ""
        main_headings = data.get("mainHeadings", {}).get("data", [])
        if main_headings:
            if isinstance(main_headings, list):
                name = main_headings[0].get("text", "")
            elif isinstance(main_headings, dict):
                name = main_headings.get("text", "")
        
        if not name:
            return None
        
        # Varianten aus x400s
        variants = []
        x400s = data.get("x400s", {}).get("x400", [])
        if isinstance(x400s, list):
            for x400 in x400s[:10]:
                if isinstance(x400, dict):
                    variant = x400.get("datafield", {}).get("subfield", {})
                    if isinstance(variant, dict):
                        variants.append(variant.get("content", ""))
        
        # Lebensdaten
        birth_date = None
        death_date = None
        
        birth_str = data.get("birthDate")
        death_str = data.get("deathDate")
        
        if birth_str:
            birth_date = self._parse_viaf_date(str(birth_str))
        if death_str:
            death_date = self._parse_viaf_date(str(death_str))
        
        return AuthorityRecord(
            source=AuthoritySourceType.VIAF,
            authority_id=viaf_id,
            name=name,
            name_variants=[v for v in variants if v],
            entity_type=entity_type,
            birth_date=birth_date,
            death_date=death_date,
            quality=QualityLevel.AUTHORITATIVE,
            raw_data=data,
        )
    
    def _parse_viaf_date(self, date_str: str) -> date | None:
        """Parst ein VIAF-Datum."""
        if not date_str:
            return None
        
        # VIAF verwendet oft nur Jahre
        try:
            year = int(date_str.strip())
            return date(year, 1, 1)
        except ValueError:
            return None


# =============================================================================
# LOC - Library of Congress Authority Files
# =============================================================================


class LOCSource(AuthoritySource):
    """
    LOC - Library of Congress Authority Files.
    
    US-amerikanischer Standard für bibliografische Normdaten.
    Sehr etabliert und zuverlässig.
    """
    
    source_type = AuthoritySourceType.LOC
    base_url = "https://id.loc.gov"
    
    async def search(self, query: str, entity_type: NodeType) -> list[AuthorityRecord]:
        """Sucht in LOC Authority Files."""
        client = await self._get_client()
        
        # Endpoint basierend auf Typ
        endpoint = {
            NodeType.PERSON: "/authorities/names",
            NodeType.ORGANIZATION: "/authorities/names",
            NodeType.LOCATION: "/authorities/subjects",
        }.get(entity_type, "/authorities/names")
        
        params = {
            "q": query,
            "format": "json",
            "count": 10,
        }
        
        response = await client.get(
            f"{self.base_url}{endpoint}/suggest2",
            params=params,
        )
        response.raise_for_status()
        
        data = response.json()
        records = []
        
        for hit in data.get("hits", []):
            # Hole Details für jeden Treffer
            uri = hit.get("uri", "")
            if uri:
                lccn = uri.split("/")[-1]
                try:
                    detail_record = await self._get_loc_detail(lccn, entity_type)
                    if detail_record:
                        records.append(detail_record)
                except Exception as e:
                    logger.debug(f"Could not fetch LOC detail for {lccn}: {e}")
        
        return records
    
    async def _get_loc_detail(self, lccn: str, entity_type: NodeType) -> AuthorityRecord | None:
        """Holt Details zu einem LOC-Eintrag."""
        client = await self._get_client()
        
        response = await client.get(
            f"{self.base_url}/authorities/names/{lccn}.json",
            follow_redirects=True,
        )
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        return self._parse_loc_record(data, lccn, entity_type)
    
    async def get_by_id(self, lccn: str) -> AuthorityRecord | None:
        """Holt einen LOC-Datensatz nach LCCN."""
        return await self._get_loc_detail(lccn, NodeType.PERSON)
    
    def _parse_loc_record(self, data: dict, lccn: str, entity_type: NodeType) -> AuthorityRecord | None:
        """Parst einen LOC-Datensatz."""
        # LOC JSON-LD Format
        graph = data if isinstance(data, list) else [data]
        
        main_entity = None
        for item in graph:
            if isinstance(item, dict) and "@type" in item:
                main_entity = item
                break
        
        if not main_entity:
            return None
        
        # Name
        name = ""
        pref_label = main_entity.get("http://www.w3.org/2004/02/skos/core#prefLabel", [])
        if pref_label:
            if isinstance(pref_label, list):
                name = pref_label[0].get("@value", "") if pref_label else ""
            elif isinstance(pref_label, dict):
                name = pref_label.get("@value", "")
        
        if not name:
            return None
        
        # Varianten
        variants = []
        alt_labels = main_entity.get("http://www.w3.org/2004/02/skos/core#altLabel", [])
        if isinstance(alt_labels, list):
            variants = [v.get("@value", "") for v in alt_labels if isinstance(v, dict)]
        
        return AuthorityRecord(
            source=AuthoritySourceType.LOC,
            authority_id=lccn,
            name=name,
            name_variants=variants,
            entity_type=entity_type,
            quality=QualityLevel.AUTHORITATIVE,
            raw_data=data,
        )


# =============================================================================
# Getty Vocabularies
# =============================================================================


class GettySource(AuthoritySource):
    """
    Getty Vocabularies - TGN, ULAN, AAT.
    
    Höchste Qualität für:
    - TGN: Geografische Namen
    - ULAN: Künstlernamen
    - AAT: Kunst & Architektur Begriffe
    """
    
    source_type = AuthoritySourceType.GETTY_ULAN
    base_url = "http://vocab.getty.edu"
    sparql_endpoint = "http://vocab.getty.edu/sparql"
    
    # Vocabulary-spezifische Endpunkte
    VOCABULARIES = {
        "tgn": ("tgn", AuthoritySourceType.GETTY_TGN),  # Geographic Names
        "ulan": ("ulan", AuthoritySourceType.GETTY_ULAN),  # Artist Names
        "aat": ("aat", AuthoritySourceType.GETTY_AAT),  # Art Terms
    }
    
    def __init__(self, vocabulary: str = "ulan"):
        super().__init__()
        self.vocabulary = vocabulary
        if vocabulary in self.VOCABULARIES:
            self.source_type = self.VOCABULARIES[vocabulary][1]
    
    async def search(self, query: str, entity_type: NodeType) -> list[AuthorityRecord]:
        """Sucht in Getty Vocabularies via SPARQL."""
        client = await self._get_client()
        
        # Wähle passendes Vocabulary
        vocab = self.vocabulary
        if entity_type == NodeType.LOCATION:
            vocab = "tgn"
        elif entity_type == NodeType.PERSON:
            vocab = "ulan"
        
        sparql_query = f"""
        PREFIX gvp: <http://vocab.getty.edu/ontology#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX xl: <http://www.w3.org/2008/05/skos-xl#>
        
        SELECT ?subject ?prefLabel ?scopeNote ?start ?end WHERE {{
            ?subject a gvp:Subject;
                     skos:inScheme <http://vocab.getty.edu/{vocab}/>;
                     gvp:prefLabelGVP/xl:literalForm ?prefLabel.
            
            FILTER(CONTAINS(LCASE(?prefLabel), LCASE("{query}")))
            
            OPTIONAL {{ ?subject gvp:scopeNote/rdf:value ?scopeNote }}
            OPTIONAL {{ ?subject gvp:estStart ?start }}
            OPTIONAL {{ ?subject gvp:estEnd ?end }}
        }}
        LIMIT 10
        """
        
        response = await client.get(
            self.sparql_endpoint,
            params={"query": sparql_query, "format": "json"},
        )
        response.raise_for_status()
        
        data = response.json()
        records = []
        
        for binding in data.get("results", {}).get("bindings", []):
            record = self._parse_getty_binding(binding, entity_type, vocab)
            if record:
                records.append(record)
        
        return records
    
    async def get_by_id(self, getty_id: str) -> AuthorityRecord | None:
        """Holt einen Getty-Datensatz nach ID."""
        client = await self._get_client()
        
        response = await client.get(
            f"{self.base_url}/{self.vocabulary}/{getty_id}.json",
            follow_redirects=True,
        )
        
        if response.status_code != 200:
            return None
        
        # Getty JSON-LD Format
        data = response.json()
        return self._parse_getty_record(data, NodeType.PERSON)
    
    def _parse_getty_binding(
        self,
        binding: dict,
        entity_type: NodeType,
        vocab: str,
    ) -> AuthorityRecord | None:
        """Parst ein SPARQL-Ergebnis."""
        subject = binding.get("subject", {}).get("value", "")
        if not subject:
            return None
        
        getty_id = subject.split("/")[-1]
        name = binding.get("prefLabel", {}).get("value", "")
        
        if not name:
            return None
        
        # Daten
        birth_date = None
        death_date = None
        
        start = binding.get("start", {}).get("value")
        end = binding.get("end", {}).get("value")
        
        if start:
            try:
                birth_date = date(int(start), 1, 1)
            except ValueError:
                pass
        
        if end:
            try:
                death_date = date(int(end), 1, 1)
            except ValueError:
                pass
        
        description = binding.get("scopeNote", {}).get("value", "")
        
        source_type = self.VOCABULARIES.get(vocab, ("ulan", AuthoritySourceType.GETTY_ULAN))[1]
        
        return AuthorityRecord(
            source=source_type,
            authority_id=getty_id,
            name=name,
            name_variants=[],
            entity_type=entity_type,
            birth_date=birth_date,
            death_date=death_date,
            description=description[:500] if description else None,
            quality=QualityLevel.VERIFIED,
        )
    
    def _parse_getty_record(self, data: dict, entity_type: NodeType) -> AuthorityRecord | None:
        """Parst einen Getty JSON-LD Record."""
        # Vereinfacht - Getty JSON-LD ist komplex
        return None


# =============================================================================
# Authority Source Manager
# =============================================================================


class AuthoritySourceManager:
    """
    Manager für alle autoritativen Datenquellen.
    
    Aggregiert Ergebnisse aus mehreren Quellen und
    dedupliziert basierend auf Authority IDs.
    """
    
    def __init__(self):
        self.sources: dict[AuthoritySourceType, AuthoritySource] = {}
    
    def register_source(self, source: AuthoritySource) -> None:
        """Registriert eine Quelle."""
        self.sources[source.source_type] = source
        logger.info(f"Registered authority source: {source.source_type}")
    
    def register_defaults(self) -> None:
        """Registriert alle Standard-Quellen."""
        self.register_source(GNDSource())
        self.register_source(VIAFSource())
        self.register_source(LOCSource())
        self.register_source(GettySource("tgn"))
        self.register_source(GettySource("ulan"))
    
    async def close_all(self) -> None:
        """Schließt alle Verbindungen."""
        for source in self.sources.values():
            await source.close()
    
    async def search_all(
        self,
        query: str,
        entity_type: NodeType,
        sources: list[AuthoritySourceType] | None = None,
    ) -> list[AuthorityImportResult]:
        """
        Durchsucht mehrere Quellen parallel.
        """
        target_sources = sources or list(self.sources.keys())
        
        tasks = []
        for source_type in target_sources:
            if source_type in self.sources:
                source = self.sources[source_type]
                tasks.append(source.import_to_graph(query, entity_type))
        
        if not tasks:
            return []
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, AuthorityImportResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Source query failed: {result}")
        
        return valid_results
    
    def merge_results(
        self,
        results: list[AuthorityImportResult],
    ) -> KnowledgeGraphExtraction:
        """
        Merged Ergebnisse aus mehreren Quellen.
        
        Priorisiert nach Quellenqualität:
        1. GND (deutsche Quellen)
        2. LOC (US Standard)
        3. VIAF (International)
        4. Getty (Spezialisiert)
        """
        # Priorisierung
        priority = {
            AuthoritySourceType.GND: 1,
            AuthoritySourceType.LOC: 2,
            AuthoritySourceType.VIAF: 3,
            AuthoritySourceType.GETTY_ULAN: 4,
            AuthoritySourceType.GETTY_TGN: 4,
        }
        
        # Sortiere nach Priorität
        sorted_results = sorted(
            results,
            key=lambda r: priority.get(r.source, 10),
        )
        
        seen_names: dict[str, Any] = {}
        all_nodes = []
        
        for result in sorted_results:
            if not result.extraction:
                continue
            
            for node in result.extraction.nodes:
                name_key = node.name.lower().strip()
                
                if name_key not in seen_names:
                    seen_names[name_key] = node
                    all_nodes.append(node)
        
        return KnowledgeGraphExtraction(
            nodes=all_nodes,
            relationships=[],
            extraction_metadata={
                "sources": [r.source.value for r in sorted_results if r.success],
                "merged": True,
            },
        )
    
    async def import_to_database(
        self,
        query: str,
        entity_type: NodeType,
        graph_manager: Any,
        sources: list[AuthoritySourceType] | None = None,
    ) -> dict[str, Any]:
        """
        Sucht und importiert Daten direkt in die Graph-Datenbank.
        """
        results = await self.search_all(query, entity_type, sources)
        merged = self.merge_results(results)
        
        if not merged.nodes:
            return {
                "success": False,
                "message": "No results found",
                "sources_queried": len(results),
            }
        
        stats = await graph_manager.add_fact_graph(merged)
        
        return {
            "success": True,
            "nodes_imported": stats.get("nodes_added", 0),
            "relationships_imported": stats.get("relationships_added", 0),
            "sources_used": [r.source.value for r in results if r.success],
            "authority_ids": [
                aid for r in results for aid in r.authority_ids
            ],
        }
