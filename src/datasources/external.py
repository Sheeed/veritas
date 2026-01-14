"""
Integration mit historischen Datenquellen.

Unterstützte Quellen:
- Wikidata (SPARQL API)
- DBpedia (SPARQL/REST)
- Wikipedia API
- Open Library (für historische Bücher)
- GeoNames (für historische Orte)

Ermöglicht das Importieren von Ground Truth Daten.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any
from urllib.parse import quote

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
# Basis-Klassen
# =============================================================================


class DataSourceType(str, Enum):
    """Unterstützte Datenquellen."""
    WIKIDATA = "wikidata"
    DBPEDIA = "dbpedia"
    WIKIPEDIA = "wikipedia"
    OPEN_LIBRARY = "open_library"
    GEONAMES = "geonames"


class ImportResult(BaseModel):
    """Ergebnis eines Datenimports."""
    source: DataSourceType
    query: str
    success: bool
    nodes_imported: int = 0
    relationships_imported: int = 0
    extraction: KnowledgeGraphExtraction | None = None
    errors: list[str] = Field(default_factory=list)
    raw_data: dict[str, Any] | None = None


class DataSourceConfig(BaseModel):
    """Konfiguration für eine Datenquelle."""
    source_type: DataSourceType
    base_url: str
    api_key: str | None = None
    rate_limit_per_minute: int = 60
    timeout_seconds: float = 30.0
    enabled: bool = True


class BaseDataSource(ABC):
    """Abstrakte Basisklasse für Datenquellen."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: datetime | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy initialization des HTTP Clients."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                headers={"User-Agent": "HistoryGuardian/1.0"},
            )
        return self._client
    
    async def close(self) -> None:
        """Schließt den HTTP Client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    @abstractmethod
    async def search_person(self, name: str, **kwargs) -> list[dict[str, Any]]:
        """Sucht nach einer Person."""
        pass
    
    @abstractmethod
    async def search_event(self, name: str, **kwargs) -> list[dict[str, Any]]:
        """Sucht nach einem historischen Ereignis."""
        pass
    
    @abstractmethod
    async def search_location(self, name: str, **kwargs) -> list[dict[str, Any]]:
        """Sucht nach einem Ort."""
        pass
    
    @abstractmethod
    async def get_entity_by_id(self, entity_id: str) -> dict[str, Any] | None:
        """Holt eine Entität nach ID."""
        pass
    
    @abstractmethod
    async def import_to_graph(
        self,
        query: str,
        entity_type: NodeType,
    ) -> ImportResult:
        """Importiert Daten als Knowledge Graph."""
        pass


# =============================================================================
# Wikidata Integration
# =============================================================================


class WikidataSource(BaseDataSource):
    """
    Integration mit Wikidata via SPARQL und REST API.
    
    Wikidata ist eine strukturierte Wissensdatenbank mit umfangreichen
    historischen Daten, Daten und Beziehungen.
    """
    
    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    API_ENDPOINT = "https://www.wikidata.org/w/api.php"
    
    # Wikidata Property IDs
    PROPS = {
        "instance_of": "P31",
        "birth_date": "P569",
        "death_date": "P570",
        "birth_place": "P19",
        "death_place": "P20",
        "occupation": "P106",
        "country_of_citizenship": "P27",
        "start_date": "P580",
        "end_date": "P582",
        "location": "P276",
        "participant": "P710",
        "part_of": "P361",
        "coordinate": "P625",
        "country": "P17",
        "inception": "P571",
        "dissolved": "P576",
        "headquarters": "P159",
    }
    
    # Wikidata Item IDs für Typen
    TYPES = {
        "human": "Q5",
        "battle": "Q178561",
        "war": "Q198",
        "treaty": "Q131569",
        "revolution": "Q10931",
        "election": "Q40231",
        "country": "Q6256",
        "city": "Q515",
        "organization": "Q43229",
    }
    
    def __init__(self, config: DataSourceConfig | None = None):
        if config is None:
            config = DataSourceConfig(
                source_type=DataSourceType.WIKIDATA,
                base_url=self.SPARQL_ENDPOINT,
            )
        super().__init__(config)
    
    async def _sparql_query(self, query: str) -> dict[str, Any]:
        """Führt eine SPARQL-Abfrage aus."""
        client = await self._get_client()
        
        response = await client.get(
            self.SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
        )
        response.raise_for_status()
        return response.json()
    
    async def _api_search(self, search_term: str, language: str = "en") -> dict[str, Any]:
        """Sucht via Wikidata API."""
        client = await self._get_client()
        
        params = {
            "action": "wbsearchentities",
            "search": search_term,
            "language": language,
            "format": "json",
            "limit": 10,
        }
        
        response = await client.get(self.API_ENDPOINT, params=params)
        response.raise_for_status()
        return response.json()
    
    async def _get_entity_data(self, qid: str) -> dict[str, Any] | None:
        """Holt detaillierte Daten zu einer Entität."""
        client = await self._get_client()
        
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "format": "json",
            "props": "labels|descriptions|claims|sitelinks",
            "languages": "en|de",
        }
        
        response = await client.get(self.API_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()
        
        return data.get("entities", {}).get(qid)
    
    def _parse_date(self, claim_value: dict) -> date | None:
        """Parst ein Wikidata-Datum."""
        try:
            time_value = claim_value.get("time", "")
            if not time_value:
                return None
            
            # Format: +YYYY-MM-DDT00:00:00Z
            time_value = time_value.lstrip("+")
            date_part = time_value.split("T")[0]
            
            parts = date_part.split("-")
            year = int(parts[0])
            month = int(parts[1]) if len(parts) > 1 and parts[1] != "00" else 1
            day = int(parts[2]) if len(parts) > 2 and parts[2] != "00" else 1
            
            return date(year, month, day)
        except (ValueError, IndexError):
            return None
    
    def _get_claim_value(
        self,
        entity: dict,
        prop_id: str,
    ) -> Any:
        """Extrahiert einen Claim-Wert aus einer Entität."""
        claims = entity.get("claims", {})
        prop_claims = claims.get(prop_id, [])
        
        if not prop_claims:
            return None
        
        # Nehme ersten Claim
        mainsnak = prop_claims[0].get("mainsnak", {})
        datavalue = mainsnak.get("datavalue", {})
        
        value_type = datavalue.get("type")
        value = datavalue.get("value")
        
        if value_type == "time":
            return self._parse_date(value)
        elif value_type == "wikibase-entityid":
            return value.get("id")
        elif value_type == "string":
            return value
        elif value_type == "quantity":
            return float(value.get("amount", 0))
        elif value_type == "globecoordinate":
            return (value.get("latitude"), value.get("longitude"))
        
        return value
    
    def _get_label(self, entity: dict, lang: str = "en") -> str:
        """Holt das Label einer Entität."""
        labels = entity.get("labels", {})
        if lang in labels:
            return labels[lang].get("value", "")
        # Fallback auf erste verfügbare Sprache
        if labels:
            return list(labels.values())[0].get("value", "")
        return ""
    
    async def search_person(self, name: str, **kwargs) -> list[dict[str, Any]]:
        """Sucht nach einer Person in Wikidata."""
        query = f"""
        SELECT ?person ?personLabel ?birth ?death ?birthPlaceLabel ?occupation
        WHERE {{
            ?person wdt:{self.PROPS['instance_of']} wd:{self.TYPES['human']}.
            ?person rdfs:label ?label.
            FILTER(CONTAINS(LCASE(?label), LCASE("{name}")))
            FILTER(LANG(?label) = "en" || LANG(?label) = "de")
            
            OPTIONAL {{ ?person wdt:{self.PROPS['birth_date']} ?birth. }}
            OPTIONAL {{ ?person wdt:{self.PROPS['death_date']} ?death. }}
            OPTIONAL {{ ?person wdt:{self.PROPS['birth_place']} ?birthPlace. }}
            OPTIONAL {{ ?person wdt:{self.PROPS['occupation']} ?occupationItem. }}
            
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,de". }}
        }}
        LIMIT 10
        """
        
        result = await self._sparql_query(query)
        return result.get("results", {}).get("bindings", [])
    
    async def search_event(self, name: str, **kwargs) -> list[dict[str, Any]]:
        """Sucht nach einem historischen Ereignis."""
        event_types = [self.TYPES["battle"], self.TYPES["war"], 
                       self.TYPES["treaty"], self.TYPES["revolution"]]
        type_filter = " ".join([f"wd:{t}" for t in event_types])
        
        query = f"""
        SELECT ?event ?eventLabel ?startDate ?endDate ?locationLabel
        WHERE {{
            VALUES ?eventType {{ {type_filter} }}
            ?event wdt:{self.PROPS['instance_of']} ?eventType.
            ?event rdfs:label ?label.
            FILTER(CONTAINS(LCASE(?label), LCASE("{name}")))
            FILTER(LANG(?label) = "en" || LANG(?label) = "de")
            
            OPTIONAL {{ ?event wdt:{self.PROPS['start_date']} ?startDate. }}
            OPTIONAL {{ ?event wdt:{self.PROPS['end_date']} ?endDate. }}
            OPTIONAL {{ ?event wdt:{self.PROPS['location']} ?location. }}
            
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,de". }}
        }}
        LIMIT 10
        """
        
        result = await self._sparql_query(query)
        return result.get("results", {}).get("bindings", [])
    
    async def search_location(self, name: str, **kwargs) -> list[dict[str, Any]]:
        """Sucht nach einem Ort."""
        query = f"""
        SELECT ?place ?placeLabel ?countryLabel ?coord
        WHERE {{
            ?place wdt:{self.PROPS['instance_of']}/wdt:P279* wd:{self.TYPES['city']}.
            ?place rdfs:label ?label.
            FILTER(CONTAINS(LCASE(?label), LCASE("{name}")))
            FILTER(LANG(?label) = "en" || LANG(?label) = "de")
            
            OPTIONAL {{ ?place wdt:{self.PROPS['country']} ?country. }}
            OPTIONAL {{ ?place wdt:{self.PROPS['coordinate']} ?coord. }}
            
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,de". }}
        }}
        LIMIT 10
        """
        
        result = await self._sparql_query(query)
        return result.get("results", {}).get("bindings", [])
    
    async def get_entity_by_id(self, entity_id: str) -> dict[str, Any] | None:
        """Holt eine Entität nach Wikidata Q-ID."""
        return await self._get_entity_data(entity_id)
    
    async def import_to_graph(
        self,
        query: str,
        entity_type: NodeType,
    ) -> ImportResult:
        """
        Importiert Wikidata-Ergebnisse als Knowledge Graph.
        """
        errors = []
        nodes = []
        relationships = []
        
        try:
            # Suche durchführen
            if entity_type == NodeType.PERSON:
                results = await self.search_person(query)
            elif entity_type == NodeType.EVENT:
                results = await self.search_event(query)
            elif entity_type == NodeType.LOCATION:
                results = await self.search_location(query)
            else:
                results = []
            
            # Ergebnisse in Nodes konvertieren
            for result in results:
                try:
                    node = self._result_to_node(result, entity_type)
                    if node:
                        nodes.append(node)
                except Exception as e:
                    errors.append(f"Failed to convert result: {e}")
            
            extraction = KnowledgeGraphExtraction(
                nodes=nodes,
                relationships=relationships,
                extraction_metadata={
                    "source": "wikidata",
                    "query": query,
                    "entity_type": entity_type.value,
                },
            )
            
            # Mark all as Facts
            for node in extraction.nodes:
                node.source_label = SourceLabel.FACT
            
            return ImportResult(
                source=DataSourceType.WIKIDATA,
                query=query,
                success=True,
                nodes_imported=len(nodes),
                relationships_imported=len(relationships),
                extraction=extraction,
                errors=errors,
            )
            
        except Exception as e:
            logger.error(f"Wikidata import failed: {e}")
            return ImportResult(
                source=DataSourceType.WIKIDATA,
                query=query,
                success=False,
                errors=[str(e)],
            )
    
    def _result_to_node(
        self,
        result: dict[str, Any],
        entity_type: NodeType,
    ) -> PersonNode | EventNode | LocationNode | None:
        """Konvertiert ein SPARQL-Ergebnis in einen Node."""
        
        def get_value(key: str) -> str | None:
            if key in result:
                return result[key].get("value")
            return None
        
        def parse_date_str(date_str: str | None) -> date | None:
            if not date_str:
                return None
            try:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
            except ValueError:
                return None
        
        if entity_type == NodeType.PERSON:
            name = get_value("personLabel")
            if not name:
                return None
            
            return PersonNode(
                name=name,
                birth_date=parse_date_str(get_value("birth")),
                death_date=parse_date_str(get_value("death")),
                source_label=SourceLabel.FACT,
                confidence=1.0,
            )
        
        elif entity_type == NodeType.EVENT:
            name = get_value("eventLabel")
            if not name:
                return None
            
            return EventNode(
                name=name,
                start_date=parse_date_str(get_value("startDate")),
                end_date=parse_date_str(get_value("endDate")),
                source_label=SourceLabel.FACT,
                confidence=1.0,
            )
        
        elif entity_type == NodeType.LOCATION:
            name = get_value("placeLabel")
            if not name:
                return None
            
            # Koordinaten parsen
            coord_str = get_value("coord")
            coordinates = None
            if coord_str:
                # Format: Point(lon lat)
                try:
                    parts = coord_str.replace("Point(", "").replace(")", "").split()
                    coordinates = (float(parts[1]), float(parts[0]))  # lat, lon
                except (ValueError, IndexError):
                    pass
            
            return LocationNode(
                name=name,
                parent_location=get_value("countryLabel"),
                coordinates=coordinates,
                source_label=SourceLabel.FACT,
                confidence=1.0,
            )
        
        return None


# =============================================================================
# DBpedia Integration
# =============================================================================


class DBpediaSource(BaseDataSource):
    """
    Integration mit DBpedia via SPARQL.
    
    DBpedia extrahiert strukturierte Daten aus Wikipedia.
    """
    
    SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
    
    def __init__(self, config: DataSourceConfig | None = None):
        if config is None:
            config = DataSourceConfig(
                source_type=DataSourceType.DBPEDIA,
                base_url=self.SPARQL_ENDPOINT,
            )
        super().__init__(config)
    
    async def _sparql_query(self, query: str) -> dict[str, Any]:
        """Führt eine SPARQL-Abfrage aus."""
        client = await self._get_client()
        
        headers = {"Accept": "application/sparql-results+json"}
        
        response = await client.get(
            self.SPARQL_ENDPOINT,
            params={"query": query},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()
    
    async def search_person(self, name: str, **kwargs) -> list[dict[str, Any]]:
        """Sucht nach einer Person in DBpedia."""
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?person ?name ?birthDate ?deathDate ?abstract
        WHERE {{
            ?person a dbo:Person.
            ?person rdfs:label ?name.
            FILTER(LANG(?name) = "en")
            FILTER(CONTAINS(LCASE(?name), LCASE("{name}")))
            
            OPTIONAL {{ ?person dbo:birthDate ?birthDate. }}
            OPTIONAL {{ ?person dbo:deathDate ?deathDate. }}
            OPTIONAL {{ 
                ?person dbo:abstract ?abstract. 
                FILTER(LANG(?abstract) = "en")
            }}
        }}
        LIMIT 10
        """
        
        result = await self._sparql_query(query)
        return result.get("results", {}).get("bindings", [])
    
    async def search_event(self, name: str, **kwargs) -> list[dict[str, Any]]:
        """Sucht nach einem Ereignis in DBpedia."""
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?event ?name ?date ?place ?abstract
        WHERE {{
            {{ ?event a dbo:Event. }}
            UNION
            {{ ?event a dbo:MilitaryConflict. }}
            
            ?event rdfs:label ?name.
            FILTER(LANG(?name) = "en")
            FILTER(CONTAINS(LCASE(?name), LCASE("{name}")))
            
            OPTIONAL {{ ?event dbo:date ?date. }}
            OPTIONAL {{ ?event dbo:place ?place. }}
            OPTIONAL {{ 
                ?event dbo:abstract ?abstract. 
                FILTER(LANG(?abstract) = "en")
            }}
        }}
        LIMIT 10
        """
        
        result = await self._sparql_query(query)
        return result.get("results", {}).get("bindings", [])
    
    async def search_location(self, name: str, **kwargs) -> list[dict[str, Any]]:
        """Sucht nach einem Ort in DBpedia."""
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
        
        SELECT ?place ?name ?country ?lat ?long
        WHERE {{
            ?place a dbo:Place.
            ?place rdfs:label ?name.
            FILTER(LANG(?name) = "en")
            FILTER(CONTAINS(LCASE(?name), LCASE("{name}")))
            
            OPTIONAL {{ ?place dbo:country ?country. }}
            OPTIONAL {{ ?place geo:lat ?lat. }}
            OPTIONAL {{ ?place geo:long ?long. }}
        }}
        LIMIT 10
        """
        
        result = await self._sparql_query(query)
        return result.get("results", {}).get("bindings", [])
    
    async def get_entity_by_id(self, entity_id: str) -> dict[str, Any] | None:
        """Holt eine Entität nach DBpedia URI."""
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        
        SELECT ?property ?value
        WHERE {{
            <{entity_id}> ?property ?value.
        }}
        LIMIT 100
        """
        
        result = await self._sparql_query(query)
        bindings = result.get("results", {}).get("bindings", [])
        
        if not bindings:
            return None
        
        entity = {"uri": entity_id, "properties": {}}
        for binding in bindings:
            prop = binding.get("property", {}).get("value", "")
            value = binding.get("value", {}).get("value", "")
            prop_name = prop.split("/")[-1].split("#")[-1]
            entity["properties"][prop_name] = value
        
        return entity
    
    async def import_to_graph(
        self,
        query: str,
        entity_type: NodeType,
    ) -> ImportResult:
        """Importiert DBpedia-Ergebnisse als Knowledge Graph."""
        # Ähnliche Implementierung wie WikidataSource
        errors = []
        nodes = []
        
        try:
            if entity_type == NodeType.PERSON:
                results = await self.search_person(query)
            elif entity_type == NodeType.EVENT:
                results = await self.search_event(query)
            elif entity_type == NodeType.LOCATION:
                results = await self.search_location(query)
            else:
                results = []
            
            for result in results:
                try:
                    node = self._result_to_node(result, entity_type)
                    if node:
                        nodes.append(node)
                except Exception as e:
                    errors.append(f"Failed to convert: {e}")
            
            extraction = KnowledgeGraphExtraction(
                nodes=nodes,
                extraction_metadata={"source": "dbpedia", "query": query},
            )
            
            for node in extraction.nodes:
                node.source_label = SourceLabel.FACT
            
            return ImportResult(
                source=DataSourceType.DBPEDIA,
                query=query,
                success=True,
                nodes_imported=len(nodes),
                extraction=extraction,
                errors=errors,
            )
            
        except Exception as e:
            logger.error(f"DBpedia import failed: {e}")
            return ImportResult(
                source=DataSourceType.DBPEDIA,
                query=query,
                success=False,
                errors=[str(e)],
            )
    
    def _result_to_node(
        self,
        result: dict[str, Any],
        entity_type: NodeType,
    ) -> PersonNode | EventNode | LocationNode | None:
        """Konvertiert DBpedia-Ergebnis in Node."""
        
        def get_value(key: str) -> str | None:
            if key in result:
                return result[key].get("value")
            return None
        
        def parse_date_str(date_str: str | None) -> date | None:
            if not date_str:
                return None
            try:
                # DBpedia verwendet oft YYYY-MM-DD Format
                return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            except ValueError:
                return None
        
        if entity_type == NodeType.PERSON:
            name = get_value("name")
            if not name:
                return None
            
            return PersonNode(
                name=name,
                birth_date=parse_date_str(get_value("birthDate")),
                death_date=parse_date_str(get_value("deathDate")),
                description=get_value("abstract")[:500] if get_value("abstract") else None,
                source_label=SourceLabel.FACT,
            )
        
        elif entity_type == NodeType.EVENT:
            name = get_value("name")
            if not name:
                return None
            
            return EventNode(
                name=name,
                start_date=parse_date_str(get_value("date")),
                description=get_value("abstract")[:500] if get_value("abstract") else None,
                source_label=SourceLabel.FACT,
            )
        
        elif entity_type == NodeType.LOCATION:
            name = get_value("name")
            if not name:
                return None
            
            lat = get_value("lat")
            long = get_value("long")
            coordinates = None
            if lat and long:
                try:
                    coordinates = (float(lat), float(long))
                except ValueError:
                    pass
            
            return LocationNode(
                name=name,
                coordinates=coordinates,
                source_label=SourceLabel.FACT,
            )
        
        return None


# =============================================================================
# Aggregierter Data Source Manager
# =============================================================================


class DataSourceManager:
    """
    Verwaltet mehrere Datenquellen und aggregiert Ergebnisse.
    
    Ermöglicht:
    - Parallele Abfragen über mehrere Quellen
    - Deduplizierung und Merging von Ergebnissen
    - Caching von Abfragen
    """
    
    def __init__(self):
        self.sources: dict[DataSourceType, BaseDataSource] = {}
        self._cache: dict[str, ImportResult] = {}
    
    def register_source(self, source: BaseDataSource) -> None:
        """Registriert eine Datenquelle."""
        self.sources[source.config.source_type] = source
        logger.info(f"Registered data source: {source.config.source_type}")
    
    def register_defaults(self) -> None:
        """Registriert Standard-Datenquellen."""
        self.register_source(WikidataSource())
        self.register_source(DBpediaSource())
    
    async def close_all(self) -> None:
        """Schließt alle Verbindungen."""
        for source in self.sources.values():
            await source.close()
    
    async def search_all(
        self,
        query: str,
        entity_type: NodeType,
        sources: list[DataSourceType] | None = None,
    ) -> list[ImportResult]:
        """
        Durchsucht alle (oder ausgewählte) Quellen parallel.
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
        
        # Fehler filtern und loggen
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Source query failed: {result}")
            elif isinstance(result, ImportResult):
                valid_results.append(result)
        
        return valid_results
    
    async def import_facts(
        self,
        queries: list[tuple[str, NodeType]],
        graph_manager: Any,  # GraphManager
        sources: list[DataSourceType] | None = None,
    ) -> dict[str, int]:
        """
        Importiert Fakten aus externen Quellen in die Graph-Datenbank.
        
        Args:
            queries: Liste von (Suchbegriff, EntityType) Tupeln
            graph_manager: GraphManager für DB-Zugriff
            sources: Optionale Liste von Quellen
            
        Returns:
            Statistiken über importierte Daten
        """
        total_nodes = 0
        total_relationships = 0
        
        for query, entity_type in queries:
            results = await self.search_all(query, entity_type, sources)
            
            for result in results:
                if result.success and result.extraction:
                    stats = await graph_manager.add_fact_graph(result.extraction)
                    total_nodes += stats.get("nodes_added", 0)
                    total_relationships += stats.get("relationships_added", 0)
        
        return {
            "total_nodes_imported": total_nodes,
            "total_relationships_imported": total_relationships,
            "queries_processed": len(queries),
        }
    
    def merge_extractions(
        self,
        results: list[ImportResult],
    ) -> KnowledgeGraphExtraction:
        """
        Merged mehrere Import-Ergebnisse zu einer Extraktion.
        
        Dedupliziert basierend auf Entity-Namen.
        """
        seen_nodes: dict[str, Any] = {}
        all_relationships = []
        
        for result in results:
            if not result.extraction:
                continue
            
            for node in result.extraction.nodes:
                key = f"{node.node_type.value}:{node.name.lower()}"
                if key not in seen_nodes:
                    seen_nodes[key] = node
                else:
                    # Merge: Behalte Node mit mehr Informationen
                    existing = seen_nodes[key]
                    if hasattr(node, 'description') and node.description:
                        if not getattr(existing, 'description', None):
                            seen_nodes[key] = node
            
            all_relationships.extend(result.extraction.relationships)
        
        return KnowledgeGraphExtraction(
            nodes=list(seen_nodes.values()),
            relationships=all_relationships,
            extraction_metadata={"merged_from": len(results)},
        )
