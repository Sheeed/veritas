"""
Authority Sources Client

Zugriff auf autoritative Datenquellen:
- GND (Deutsche Nationalbibliothek via lobid.org)
- VIAF (Virtual International Authority File)
- LOC (Library of Congress)
- Wikidata (strukturierte Daten, NICHT Wikipedia)
- Bundesarchiv (Bestaende-Suche)
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


class AuthorityType(str, Enum):
    """Typen von Authority Records."""
    PERSON = "person"
    PLACE = "place"
    EVENT = "event"
    ORGANIZATION = "organization"
    WORK = "work"
    CONCEPT = "concept"


@dataclass
class AuthorityRecord:
    """Ein Eintrag aus einer autoritativen Quelle."""
    
    source: str                    # GND, VIAF, LOC, WIKIDATA
    id: str                        # ID in der Quelle
    uri: str                       # Vollstaendige URI
    label: str                     # Hauptbezeichnung
    type: AuthorityType
    description: Optional[str] = None
    birth_date: Optional[str] = None
    death_date: Optional[str] = None
    alternate_names: list[str] = None
    related_ids: dict[str, str] = None  # Andere Authority IDs
    raw_data: dict = None
    
    def __post_init__(self):
        if self.alternate_names is None:
            self.alternate_names = []
        if self.related_ids is None:
            self.related_ids = {}


@dataclass 
class FactRecord:
    """Ein verifizierter Fakt aus einer autoritativen Quelle."""
    
    source: str
    source_uri: str
    claim: str
    value: Any
    confidence: float = 1.0
    date_verified: Optional[str] = None
    supporting_refs: list[str] = None
    
    def __post_init__(self):
        if self.supporting_refs is None:
            self.supporting_refs = []


class AuthorityClient:
    """
    Client fuer autoritative Datenquellen.
    
    Unterstuetzt:
    - GND (lobid.org API)
    - VIAF (viaf.org API)
    - LOC (id.loc.gov API)
    - Wikidata (wikidata.org API)
    """
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.base_urls = {
            "gnd": "https://lobid.org/gnd",
            "viaf": "https://viaf.org/viaf",
            "loc": "https://id.loc.gov/authorities",
            "wikidata": "https://www.wikidata.org/w/api.php",
        }
    
    # =========================================================================
    # GND (Deutsche Nationalbibliothek)
    # =========================================================================
    
    async def search_gnd(
        self, 
        query: str, 
        entity_type: Optional[AuthorityType] = None,
        limit: int = 10
    ) -> list[AuthorityRecord]:
        """
        Sucht in der GND via lobid.org API.
        
        Args:
            query: Suchbegriff
            entity_type: Optional - Person, Place, etc.
            limit: Max Ergebnisse
        
        Returns:
            Liste von AuthorityRecords
        """
        type_filter = ""
        if entity_type == AuthorityType.PERSON:
            type_filter = "&filter=type:Person"
        elif entity_type == AuthorityType.PLACE:
            type_filter = "&filter=type:PlaceOrGeographicName"
        elif entity_type == AuthorityType.EVENT:
            type_filter = "&filter=type:SubjectHeading"
        elif entity_type == AuthorityType.ORGANIZATION:
            type_filter = "&filter=type:CorporateBody"
        
        url = f"{self.base_urls['gnd']}/search?q={quote(query)}&size={limit}&format=json{type_filter}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
            
            records = []
            for item in data.get("member", []):
                gnd_id = item.get("gndIdentifier", "")
                
                # Typ bestimmen
                item_type = AuthorityType.CONCEPT
                type_list = item.get("type", [])
                if "Person" in type_list:
                    item_type = AuthorityType.PERSON
                elif "PlaceOrGeographicName" in type_list:
                    item_type = AuthorityType.PLACE
                elif "CorporateBody" in type_list:
                    item_type = AuthorityType.ORGANIZATION
                
                # Lebensdaten
                birth = None
                death = None
                if "dateOfBirth" in item:
                    birth = item["dateOfBirth"][0] if isinstance(item["dateOfBirth"], list) else item["dateOfBirth"]
                if "dateOfDeath" in item:
                    death = item["dateOfDeath"][0] if isinstance(item["dateOfDeath"], list) else item["dateOfDeath"]
                
                record = AuthorityRecord(
                    source="GND",
                    id=gnd_id,
                    uri=f"https://d-nb.info/gnd/{gnd_id}",
                    label=item.get("preferredName", ""),
                    type=item_type,
                    description=item.get("biographicalOrHistoricalInformation", [None])[0] if item.get("biographicalOrHistoricalInformation") else None,
                    birth_date=birth,
                    death_date=death,
                    alternate_names=item.get("variantName", []),
                    raw_data=item,
                )
                records.append(record)
            
            return records
        
        except Exception as e:
            logger.error(f"GND search failed: {e}")
            return []
    
    async def get_gnd(self, gnd_id: str) -> Optional[AuthorityRecord]:
        """Holt einen GND-Eintrag direkt."""
        url = f"{self.base_urls['gnd']}/{gnd_id}.json"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                item = response.json()
            
            item_type = AuthorityType.CONCEPT
            type_list = item.get("type", [])
            if "Person" in type_list:
                item_type = AuthorityType.PERSON
            elif "PlaceOrGeographicName" in type_list:
                item_type = AuthorityType.PLACE
            
            return AuthorityRecord(
                source="GND",
                id=gnd_id,
                uri=f"https://d-nb.info/gnd/{gnd_id}",
                label=item.get("preferredName", ""),
                type=item_type,
                description=item.get("biographicalOrHistoricalInformation", [None])[0] if item.get("biographicalOrHistoricalInformation") else None,
                birth_date=item.get("dateOfBirth", [None])[0] if item.get("dateOfBirth") else None,
                death_date=item.get("dateOfDeath", [None])[0] if item.get("dateOfDeath") else None,
                alternate_names=item.get("variantName", []),
                related_ids={"viaf": item.get("sameAs", [{}])[0].get("id", "")} if item.get("sameAs") else {},
                raw_data=item,
            )
        
        except Exception as e:
            logger.error(f"GND get failed: {e}")
            return None
    
    # =========================================================================
    # VIAF (Virtual International Authority File)
    # =========================================================================
    
    async def search_viaf(
        self,
        query: str,
        entity_type: Optional[AuthorityType] = None,
        limit: int = 10
    ) -> list[AuthorityRecord]:
        """Sucht in VIAF."""
        
        # VIAF Query-Typen
        cql_index = "local.mainHeadingEl"
        if entity_type == AuthorityType.PERSON:
            cql_index = "local.personalNames"
        elif entity_type == AuthorityType.ORGANIZATION:
            cql_index = "local.corporateNames"
        elif entity_type == AuthorityType.PLACE:
            cql_index = "local.geographicNames"
        
        url = f"{self.base_urls['viaf']}/search?query={cql_index}+all+%22{quote(query)}%22&maximumRecords={limit}&httpAccept=application/json"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
            
            records = []
            search_results = data.get("searchRetrieveResponse", {}).get("records", [])
            
            for item in search_results:
                record_data = item.get("record", {}).get("recordData", {})
                viaf_id = record_data.get("viafID", "")
                
                # Hauptname extrahieren
                main_heading = record_data.get("mainHeadings", {}).get("data", [])
                if isinstance(main_heading, list) and main_heading:
                    label = main_heading[0].get("text", "")
                elif isinstance(main_heading, dict):
                    label = main_heading.get("text", "")
                else:
                    label = str(main_heading)
                
                # Lebensdaten
                birth = record_data.get("birthDate", "")
                death = record_data.get("deathDate", "")
                
                record = AuthorityRecord(
                    source="VIAF",
                    id=viaf_id,
                    uri=f"https://viaf.org/viaf/{viaf_id}",
                    label=label,
                    type=entity_type or AuthorityType.CONCEPT,
                    birth_date=birth if birth else None,
                    death_date=death if death else None,
                    raw_data=record_data,
                )
                records.append(record)
            
            return records
        
        except Exception as e:
            logger.error(f"VIAF search failed: {e}")
            return []
    
    # =========================================================================
    # Wikidata (Strukturierte Daten)
    # =========================================================================
    
    async def search_wikidata(
        self,
        query: str,
        entity_type: Optional[AuthorityType] = None,
        limit: int = 10,
        language: str = "de"
    ) -> list[AuthorityRecord]:
        """
        Sucht in Wikidata (NICHT Wikipedia!).
        
        Wikidata enthaelt strukturierte, verifizierte Fakten.
        """
        url = f"{self.base_urls['wikidata']}?action=wbsearchentities&search={quote(query)}&language={language}&limit={limit}&format=json"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
            
            records = []
            for item in data.get("search", []):
                record = AuthorityRecord(
                    source="WIKIDATA",
                    id=item.get("id", ""),
                    uri=item.get("concepturi", ""),
                    label=item.get("label", ""),
                    type=entity_type or AuthorityType.CONCEPT,
                    description=item.get("description", ""),
                    raw_data=item,
                )
                records.append(record)
            
            return records
        
        except Exception as e:
            logger.error(f"Wikidata search failed: {e}")
            return []
    
    async def get_wikidata_facts(self, wikidata_id: str) -> list[FactRecord]:
        """
        Holt verifizierte Fakten aus Wikidata.
        
        Beispiel: Q517 (Napoleon) -> Geburtsdatum, Groesse, Todesort, etc.
        """
        url = f"{self.base_urls['wikidata']}?action=wbgetentities&ids={wikidata_id}&format=json&props=claims|labels|descriptions"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
            
            entity = data.get("entities", {}).get(wikidata_id, {})
            claims = entity.get("claims", {})
            
            facts = []
            
            # Wichtige Properties extrahieren
            property_map = {
                "P569": "Geburtsdatum",
                "P570": "Sterbedatum",
                "P19": "Geburtsort",
                "P20": "Sterbeort",
                "P2048": "Koerpergroesse",
                "P27": "Staatsangehoerigkeit",
                "P106": "Beruf",
                "P39": "Amt",
            }
            
            for prop_id, prop_name in property_map.items():
                if prop_id in claims:
                    claim_data = claims[prop_id][0]
                    mainsnak = claim_data.get("mainsnak", {})
                    datavalue = mainsnak.get("datavalue", {})
                    
                    # Wert extrahieren
                    value = None
                    if datavalue.get("type") == "time":
                        value = datavalue.get("value", {}).get("time", "")
                    elif datavalue.get("type") == "quantity":
                        amount = datavalue.get("value", {}).get("amount", "")
                        unit = datavalue.get("value", {}).get("unit", "").split("/")[-1]
                        value = f"{amount} ({unit})"
                    elif datavalue.get("type") == "wikibase-entityid":
                        value = datavalue.get("value", {}).get("id", "")
                    else:
                        value = str(datavalue.get("value", ""))
                    
                    if value:
                        # Referenzen
                        refs = []
                        for ref in claim_data.get("references", []):
                            for snak in ref.get("snaks", {}).get("P248", []):
                                ref_id = snak.get("datavalue", {}).get("value", {}).get("id", "")
                                if ref_id:
                                    refs.append(f"https://www.wikidata.org/wiki/{ref_id}")
                        
                        facts.append(FactRecord(
                            source="WIKIDATA",
                            source_uri=f"https://www.wikidata.org/wiki/{wikidata_id}#{prop_id}",
                            claim=prop_name,
                            value=value,
                            confidence=1.0 if refs else 0.8,
                            supporting_refs=refs,
                        ))
            
            return facts
        
        except Exception as e:
            logger.error(f"Wikidata facts failed: {e}")
            return []
    
    # =========================================================================
    # Library of Congress
    # =========================================================================
    
    async def search_loc(
        self,
        query: str,
        entity_type: Optional[AuthorityType] = None,
        limit: int = 10
    ) -> list[AuthorityRecord]:
        """Sucht in der Library of Congress."""
        
        # LOC Endpoints
        endpoint = "names"
        if entity_type == AuthorityType.PLACE:
            endpoint = "subjects"
        elif entity_type == AuthorityType.CONCEPT:
            endpoint = "subjects"
        
        url = f"{self.base_urls['loc']}/{endpoint}/suggest2/?q={quote(query)}&count={limit}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
            
            records = []
            for item in data.get("hits", []):
                loc_uri = item.get("uri", "")
                loc_id = loc_uri.split("/")[-1] if loc_uri else ""
                
                record = AuthorityRecord(
                    source="LOC",
                    id=loc_id,
                    uri=loc_uri,
                    label=item.get("suggestLabel", item.get("aLabel", "")),
                    type=entity_type or AuthorityType.CONCEPT,
                    raw_data=item,
                )
                records.append(record)
            
            return records
        
        except Exception as e:
            logger.error(f"LOC search failed: {e}")
            return []
    
    # =========================================================================
    # Unified Search
    # =========================================================================
    
    async def search_all(
        self,
        query: str,
        entity_type: Optional[AuthorityType] = None,
        sources: list[str] = None,
        limit_per_source: int = 5
    ) -> dict[str, list[AuthorityRecord]]:
        """
        Sucht parallel in allen aktivierten Quellen.
        
        Args:
            query: Suchbegriff
            entity_type: Optional Filter
            sources: Liste der Quellen ["gnd", "viaf", "wikidata", "loc"]
            limit_per_source: Max Ergebnisse pro Quelle
        
        Returns:
            Dict mit Ergebnissen pro Quelle
        """
        if sources is None:
            sources = ["gnd", "viaf", "wikidata", "loc"]
        
        tasks = []
        source_names = []
        
        if "gnd" in sources:
            tasks.append(self.search_gnd(query, entity_type, limit_per_source))
            source_names.append("gnd")
        
        if "viaf" in sources:
            tasks.append(self.search_viaf(query, entity_type, limit_per_source))
            source_names.append("viaf")
        
        if "wikidata" in sources:
            tasks.append(self.search_wikidata(query, entity_type, limit_per_source))
            source_names.append("wikidata")
        
        if "loc" in sources:
            tasks.append(self.search_loc(query, entity_type, limit_per_source))
            source_names.append("loc")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for name, result in zip(source_names, results):
            if isinstance(result, Exception):
                logger.error(f"Search failed for {name}: {result}")
                output[name] = []
            else:
                output[name] = result
        
        return output
    
    async def verify_person_fact(
        self,
        person_name: str,
        claim_type: str,
        claimed_value: Any
    ) -> dict:
        """
        Verifiziert einen Fakt ueber eine Person gegen autoritative Quellen.
        
        Args:
            person_name: Name der Person
            claim_type: Art des Claims (birth_date, death_date, height, etc.)
            claimed_value: Der behauptete Wert
        
        Returns:
            Verifikationsergebnis mit Quellen
        """
        result = {
            "verified": False,
            "claimed": claimed_value,
            "actual": None,
            "sources": [],
            "confidence": 0.0,
        }
        
        # In Wikidata suchen
        wikidata_results = await self.search_wikidata(person_name, AuthorityType.PERSON, 1)
        
        if wikidata_results:
            wikidata_id = wikidata_results[0].id
            facts = await self.get_wikidata_facts(wikidata_id)
            
            # Passenden Fakt finden
            claim_map = {
                "birth_date": "Geburtsdatum",
                "death_date": "Sterbedatum",
                "height": "Koerpergroesse",
                "birthplace": "Geburtsort",
            }
            
            target_claim = claim_map.get(claim_type, claim_type)
            
            for fact in facts:
                if fact.claim == target_claim:
                    result["actual"] = fact.value
                    result["sources"].append({
                        "source": fact.source,
                        "uri": fact.source_uri,
                        "refs": fact.supporting_refs,
                    })
                    result["confidence"] = fact.confidence
                    
                    # Einfacher Vergleich (kann verbessert werden)
                    if str(claimed_value).lower() in str(fact.value).lower():
                        result["verified"] = True
                    break
        
        # GND als Backup
        if not result["actual"]:
            gnd_results = await self.search_gnd(person_name, AuthorityType.PERSON, 1)
            if gnd_results:
                gnd_record = gnd_results[0]
                if claim_type == "birth_date" and gnd_record.birth_date:
                    result["actual"] = gnd_record.birth_date
                    result["sources"].append({
                        "source": "GND",
                        "uri": gnd_record.uri,
                    })
                    result["confidence"] = 0.9
        
        return result


# =============================================================================
# Singleton
# =============================================================================

_client_instance: Optional[AuthorityClient] = None


def get_authority_client() -> AuthorityClient:
    """Gibt die Client-Instanz zurueck."""
    global _client_instance
    if _client_instance is None:
        _client_instance = AuthorityClient()
    return _client_instance
