"""
Veritas Authoritative Source Verifier

Verifiziert historische Behauptungen NUR gegen autoritative Quellen.

KEINE Wikipedia/Wikidata - diese sind keine zuverlässigen Primärquellen!

Quellen-Hierarchie (nach Zuverlässigkeit):
1. Authority Files: GND, VIAF, LOC (höchste Zuverlässigkeit)
2. Fact-Check Organizations: Google Fact Check, ClaimBuster
3. Academic Sources: CrossRef (DOI-basiert)
4. Encyclopedia: Britannica (redaktionell geprüft)

Features:
- Strikte Quellen-Hierarchie
- Nur verifizierte, autoritative Quellen
- Transparente Zuverlässigkeits-Scores
- Caching für Performance
"""

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import quote, urlencode
from enum import Enum

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Source Reliability Tiers
# =============================================================================

class SourceTier(str, Enum):
    """Quellen-Zuverlässigkeits-Stufen."""
    
    TIER_1_AUTHORITY = "tier_1_authority"      # GND, VIAF, LOC - 95%
    TIER_2_FACTCHECK = "tier_2_factcheck"      # Google Fact Check, Snopes - 85%
    TIER_3_ACADEMIC = "tier_3_academic"        # CrossRef, DOI - 80%
    TIER_4_ENCYCLOPEDIA = "tier_4_encyclopedia" # Britannica - 75%


TIER_RELIABILITY = {
    SourceTier.TIER_1_AUTHORITY: 0.95,
    SourceTier.TIER_2_FACTCHECK: 0.85,
    SourceTier.TIER_3_ACADEMIC: 0.80,
    SourceTier.TIER_4_ENCYCLOPEDIA: 0.75,
}


# =============================================================================
# Models
# =============================================================================

class AuthoritativeSourceResult(BaseModel):
    """Ergebnis einer autoritativen Quellenabfrage."""
    
    source_name: str
    source_type: str
    tier: SourceTier
    
    found: bool = False
    
    title: Optional[str] = None
    url: Optional[str] = None
    extract: Optional[str] = None
    
    # Scores
    relevance_score: float = 0.0
    reliability_score: float = 0.0  # Basierend auf Tier
    
    # Fact-Check spezifisch
    claim_review: Optional[str] = None  # Bewertung des Fact-Checkers
    rating: Optional[str] = None  # z.B. "False", "Mostly False", etc.
    
    raw_data: Optional[Dict] = None
    error: Optional[str] = None


class AuthoritativeVerificationResult(BaseModel):
    """Aggregiertes Verifikationsergebnis von autoritativen Quellen."""
    
    query: str
    verified: bool = False
    confidence: float = 0.0
    
    # Quellen-Stats
    sources_checked: int = 0
    sources_found: int = 0
    highest_tier_found: Optional[SourceTier] = None
    
    results: List[AuthoritativeSourceResult] = Field(default_factory=list)
    
    # Fact-Check Ergebnis
    fact_check_rating: Optional[str] = None
    fact_check_consensus: Optional[str] = None
    
    summary: str = ""
    recommendation: str = ""
    
    cached: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Authority Files Client (GND, VIAF, LOC)
# =============================================================================

class AuthorityFilesClient:
    """
    Client für Authority Files (höchste Zuverlässigkeit).
    
    - GND (Deutsche Nationalbibliothek)
    - VIAF (Virtual International Authority File)
    - LOC (Library of Congress)
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)
    
    async def search_gnd(self, query: str) -> AuthoritativeSourceResult:
        """Sucht in der GND (Deutsche Nationalbibliothek)."""
        url = "https://lobid.org/gnd/search"
        
        params = {
            "q": query,
            "format": "json",
            "size": 5,
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            members = data.get("member", [])
            if not members:
                return AuthoritativeSourceResult(
                    source_name="GND (Deutsche Nationalbibliothek)",
                    source_type="authority_file",
                    tier=SourceTier.TIER_1_AUTHORITY,
                    found=False,
                    reliability_score=TIER_RELIABILITY[SourceTier.TIER_1_AUTHORITY],
                )
            
            best = members[0]
            
            return AuthoritativeSourceResult(
                source_name="GND (Deutsche Nationalbibliothek)",
                source_type="authority_file",
                tier=SourceTier.TIER_1_AUTHORITY,
                found=True,
                title=best.get("preferredName", ""),
                url=best.get("id", ""),
                extract=self._format_gnd_info(best),
                relevance_score=self._calc_relevance(query, best.get("preferredName", "")),
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_1_AUTHORITY],
                raw_data=best,
            )
            
        except Exception as e:
            logger.error(f"GND search failed: {e}")
            return AuthoritativeSourceResult(
                source_name="GND",
                source_type="authority_file",
                tier=SourceTier.TIER_1_AUTHORITY,
                found=False,
                error=str(e),
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_1_AUTHORITY],
            )
    
    def _format_gnd_info(self, data: Dict) -> str:
        """Formatiert GND-Daten als lesbaren Text."""
        parts = []
        
        if data.get("preferredName"):
            parts.append(f"Name: {data['preferredName']}")
        if data.get("biographicalOrHistoricalInformation"):
            info = data["biographicalOrHistoricalInformation"]
            if isinstance(info, list):
                parts.append(f"Info: {info[0]}")
            else:
                parts.append(f"Info: {info}")
        if data.get("dateOfBirth"):
            parts.append(f"Born: {data['dateOfBirth']}")
        if data.get("dateOfDeath"):
            parts.append(f"Died: {data['dateOfDeath']}")
        if data.get("professionOrOccupation"):
            occupations = data["professionOrOccupation"]
            if isinstance(occupations, list) and occupations:
                occ_labels = [o.get("label", str(o)) if isinstance(o, dict) else str(o) for o in occupations[:3]]
                parts.append(f"Occupation: {', '.join(occ_labels)}")
        
        return " | ".join(parts) if parts else "No additional information"
    
    async def search_viaf(self, query: str) -> AuthoritativeSourceResult:
        """Sucht im VIAF (Virtual International Authority File)."""
        url = "https://viaf.org/viaf/search"
        
        params = {
            "query": f'local.names all "{query}"',
            "httpAccept": "application/json",
            "maximumRecords": 5,
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            records = data.get("searchRetrieveResponse", {}).get("records", [])
            if not records:
                return AuthoritativeSourceResult(
                    source_name="VIAF (Virtual International Authority File)",
                    source_type="authority_file",
                    tier=SourceTier.TIER_1_AUTHORITY,
                    found=False,
                    reliability_score=TIER_RELIABILITY[SourceTier.TIER_1_AUTHORITY],
                )
            
            record = records[0].get("record", {}).get("recordData", {})
            viaf_id = record.get("viafID", "")
            
            # Namen extrahieren
            main_headings = record.get("mainHeadings", {}).get("data", [])
            if isinstance(main_headings, dict):
                main_headings = [main_headings]
            
            name = ""
            if main_headings:
                first = main_headings[0]
                name = first.get("text", "") if isinstance(first, dict) else str(first)
            
            return AuthoritativeSourceResult(
                source_name="VIAF (Virtual International Authority File)",
                source_type="authority_file",
                tier=SourceTier.TIER_1_AUTHORITY,
                found=True,
                title=name,
                url=f"https://viaf.org/viaf/{viaf_id}" if viaf_id else None,
                extract=f"VIAF ID: {viaf_id}",
                relevance_score=self._calc_relevance(query, name),
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_1_AUTHORITY],
                raw_data=record,
            )
            
        except Exception as e:
            logger.error(f"VIAF search failed: {e}")
            return AuthoritativeSourceResult(
                source_name="VIAF",
                source_type="authority_file",
                tier=SourceTier.TIER_1_AUTHORITY,
                found=False,
                error=str(e),
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_1_AUTHORITY],
            )
    
    async def search_loc(self, query: str) -> AuthoritativeSourceResult:
        """Sucht in der Library of Congress."""
        url = "https://id.loc.gov/search/"
        
        params = {
            "q": query,
            "format": "json",
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = data if isinstance(data, list) else []
            
            # Filtern auf relevante Ergebnisse
            relevant = [r for r in results if isinstance(r, dict) and r.get("@id")]
            
            if not relevant:
                return AuthoritativeSourceResult(
                    source_name="Library of Congress",
                    source_type="authority_file",
                    tier=SourceTier.TIER_1_AUTHORITY,
                    found=False,
                    reliability_score=TIER_RELIABILITY[SourceTier.TIER_1_AUTHORITY],
                )
            
            best = relevant[0]
            
            return AuthoritativeSourceResult(
                source_name="Library of Congress",
                source_type="authority_file",
                tier=SourceTier.TIER_1_AUTHORITY,
                found=True,
                title=best.get("aLabel", best.get("@id", "")),
                url=best.get("@id"),
                relevance_score=self._calc_relevance(query, best.get("aLabel", "")),
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_1_AUTHORITY],
                raw_data=best,
            )
            
        except Exception as e:
            logger.error(f"LOC search failed: {e}")
            return AuthoritativeSourceResult(
                source_name="Library of Congress",
                source_type="authority_file",
                tier=SourceTier.TIER_1_AUTHORITY,
                found=False,
                error=str(e),
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_1_AUTHORITY],
            )
    
    def _calc_relevance(self, query: str, result: str) -> float:
        """Berechnet Relevanz."""
        if not result:
            return 0.0
        
        query_words = set(query.lower().split())
        result_words = set(result.lower().split())
        
        intersection = len(query_words & result_words)
        if len(query_words) == 0:
            return 0.0
        
        return min(1.0, intersection / len(query_words))
    
    async def close(self):
        await self.client.aclose()


# =============================================================================
# Fact-Check APIs Client
# =============================================================================

class FactCheckClient:
    """
    Client für Fact-Checking Organisationen.
    
    - Google Fact Check Tools API
    - ClaimBuster API
    """
    
    def __init__(self, google_api_key: Optional[str] = None):
        self.client = httpx.AsyncClient(timeout=15.0)
        self.google_api_key = google_api_key or self._get_google_key()
    
    def _get_google_key(self) -> Optional[str]:
        """Holt Google API Key aus Umgebung."""
        import os
        return os.getenv("GOOGLE_FACTCHECK_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    async def search_google_factcheck(self, query: str) -> AuthoritativeSourceResult:
        """
        Sucht in Google Fact Check Tools API.
        
        Diese API aggregiert Fact-Checks von verifizierten Organisationen wie:
        - Snopes
        - PolitiFact
        - FactCheck.org
        - AFP Fact Check
        - Reuters Fact Check
        - etc.
        """
        if not self.google_api_key:
            return AuthoritativeSourceResult(
                source_name="Google Fact Check",
                source_type="fact_check",
                tier=SourceTier.TIER_2_FACTCHECK,
                found=False,
                error="API key not configured",
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_2_FACTCHECK],
            )
        
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        
        params = {
            "key": self.google_api_key,
            "query": query,
            "languageCode": "en",
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            claims = data.get("claims", [])
            if not claims:
                return AuthoritativeSourceResult(
                    source_name="Google Fact Check",
                    source_type="fact_check",
                    tier=SourceTier.TIER_2_FACTCHECK,
                    found=False,
                    reliability_score=TIER_RELIABILITY[SourceTier.TIER_2_FACTCHECK],
                )
            
            # Beste Übereinstimmung
            claim = claims[0]
            claim_review = claim.get("claimReview", [{}])[0] if claim.get("claimReview") else {}
            
            publisher = claim_review.get("publisher", {}).get("name", "Unknown")
            rating = claim_review.get("textualRating", "")
            
            return AuthoritativeSourceResult(
                source_name=f"Fact Check ({publisher})",
                source_type="fact_check",
                tier=SourceTier.TIER_2_FACTCHECK,
                found=True,
                title=claim.get("text", ""),
                url=claim_review.get("url"),
                extract=claim_review.get("title", ""),
                claim_review=claim_review.get("title"),
                rating=rating,
                relevance_score=0.9,  # Fact-Checks sind direkt relevant
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_2_FACTCHECK],
                raw_data=claim,
            )
            
        except Exception as e:
            logger.error(f"Google Fact Check failed: {e}")
            return AuthoritativeSourceResult(
                source_name="Google Fact Check",
                source_type="fact_check",
                tier=SourceTier.TIER_2_FACTCHECK,
                found=False,
                error=str(e),
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_2_FACTCHECK],
            )
    
    async def check_claimbuster(self, query: str) -> AuthoritativeSourceResult:
        """
        Nutzt ClaimBuster API für Claim-Scoring.
        
        ClaimBuster bewertet wie "check-worthy" eine Behauptung ist.
        """
        url = "https://idir.uta.edu/claimbuster/api/v2/score/text/"
        
        try:
            response = await self.client.post(
                url,
                json={"input_text": query},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            if not results:
                return AuthoritativeSourceResult(
                    source_name="ClaimBuster",
                    source_type="fact_check",
                    tier=SourceTier.TIER_2_FACTCHECK,
                    found=False,
                    reliability_score=TIER_RELIABILITY[SourceTier.TIER_2_FACTCHECK],
                )
            
            # Höchster Score
            best = max(results, key=lambda x: x.get("score", 0))
            score = best.get("score", 0)
            
            # Interpretation
            if score > 0.7:
                interpretation = "Highly check-worthy claim"
            elif score > 0.5:
                interpretation = "Moderately check-worthy claim"
            else:
                interpretation = "Low check-worthiness"
            
            return AuthoritativeSourceResult(
                source_name="ClaimBuster",
                source_type="fact_check",
                tier=SourceTier.TIER_2_FACTCHECK,
                found=True,
                title=best.get("text", query),
                extract=f"Check-worthiness score: {score:.2f} - {interpretation}",
                relevance_score=score,
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_2_FACTCHECK],
                raw_data=best,
            )
            
        except Exception as e:
            logger.error(f"ClaimBuster failed: {e}")
            return AuthoritativeSourceResult(
                source_name="ClaimBuster",
                source_type="fact_check",
                tier=SourceTier.TIER_2_FACTCHECK,
                found=False,
                error=str(e),
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_2_FACTCHECK],
            )
    
    async def close(self):
        await self.client.aclose()


# =============================================================================
# Academic Sources Client
# =============================================================================

class AcademicSourcesClient:
    """
    Client für akademische Quellen.
    
    - CrossRef (DOI-basierte Suche)
    - Open Library
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=15.0)
    
    async def search_crossref(self, query: str) -> AuthoritativeSourceResult:
        """
        Sucht in CrossRef nach wissenschaftlichen Publikationen.
        
        CrossRef ist die offizielle DOI-Registrierungsagentur.
        """
        url = "https://api.crossref.org/works"
        
        params = {
            "query": query,
            "rows": 5,
        }
        
        try:
            response = await self.client.get(
                url, 
                params=params,
                headers={"User-Agent": "HistoryGuardian/1.0 (mailto:contact@example.com)"}
            )
            response.raise_for_status()
            data = response.json()
            
            items = data.get("message", {}).get("items", [])
            if not items:
                return AuthoritativeSourceResult(
                    source_name="CrossRef (Academic)",
                    source_type="academic",
                    tier=SourceTier.TIER_3_ACADEMIC,
                    found=False,
                    reliability_score=TIER_RELIABILITY[SourceTier.TIER_3_ACADEMIC],
                )
            
            best = items[0]
            
            # Titel extrahieren
            titles = best.get("title", [])
            title = titles[0] if titles else "No title"
            
            # Autoren
            authors = best.get("author", [])
            author_str = ", ".join([
                f"{a.get('given', '')} {a.get('family', '')}" 
                for a in authors[:3]
            ]) if authors else "Unknown"
            
            # DOI URL
            doi = best.get("DOI", "")
            url = f"https://doi.org/{doi}" if doi else None
            
            return AuthoritativeSourceResult(
                source_name="CrossRef (Academic)",
                source_type="academic",
                tier=SourceTier.TIER_3_ACADEMIC,
                found=True,
                title=title,
                url=url,
                extract=f"Authors: {author_str} | Published: {best.get('published-print', {}).get('date-parts', [['']])[0][0] or 'Unknown'}",
                relevance_score=best.get("score", 0) / 100,  # CrossRef score normalisieren
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_3_ACADEMIC],
                raw_data=best,
            )
            
        except Exception as e:
            logger.error(f"CrossRef search failed: {e}")
            return AuthoritativeSourceResult(
                source_name="CrossRef",
                source_type="academic",
                tier=SourceTier.TIER_3_ACADEMIC,
                found=False,
                error=str(e),
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_3_ACADEMIC],
            )
    
    async def search_open_library(self, query: str) -> AuthoritativeSourceResult:
        """
        Sucht in Open Library nach Büchern.
        """
        url = "https://openlibrary.org/search.json"
        
        params = {
            "q": query,
            "limit": 5,
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            docs = data.get("docs", [])
            if not docs:
                return AuthoritativeSourceResult(
                    source_name="Open Library",
                    source_type="academic",
                    tier=SourceTier.TIER_3_ACADEMIC,
                    found=False,
                    reliability_score=TIER_RELIABILITY[SourceTier.TIER_3_ACADEMIC],
                )
            
            best = docs[0]
            
            return AuthoritativeSourceResult(
                source_name="Open Library",
                source_type="academic",
                tier=SourceTier.TIER_3_ACADEMIC,
                found=True,
                title=best.get("title", ""),
                url=f"https://openlibrary.org{best.get('key', '')}" if best.get("key") else None,
                extract=f"Author: {', '.join(best.get('author_name', ['Unknown'])[:3])} | Year: {best.get('first_publish_year', 'Unknown')}",
                relevance_score=0.7,
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_3_ACADEMIC],
                raw_data=best,
            )
            
        except Exception as e:
            logger.error(f"Open Library search failed: {e}")
            return AuthoritativeSourceResult(
                source_name="Open Library",
                source_type="academic",
                tier=SourceTier.TIER_3_ACADEMIC,
                found=False,
                error=str(e),
                reliability_score=TIER_RELIABILITY[SourceTier.TIER_3_ACADEMIC],
            )
    
    async def close(self):
        await self.client.aclose()


# =============================================================================
# Cache
# =============================================================================

class VerificationCache:
    """File-basierter Cache für Verifikationsergebnisse."""
    
    def __init__(self, cache_dir: str = "data/cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_key(self, query: str) -> str:
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[AuthoritativeVerificationResult]:
        key = self._get_key(query)
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            timestamp = datetime.fromisoformat(data.get("timestamp", "2000-01-01"))
            if datetime.now() - timestamp > self.ttl:
                cache_file.unlink()
                return None
            
            result = AuthoritativeVerificationResult(**data)
            result.cached = True
            return result
            
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            return None
    
    def set(self, query: str, result: AuthoritativeVerificationResult) -> None:
        key = self._get_key(query)
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            data = result.model_dump()
            data["timestamp"] = datetime.now().isoformat()
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    def clear(self) -> int:
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count


# =============================================================================
# Main Authoritative Source Verifier
# =============================================================================

class AuthoritativeSourceVerifier:
    """
    Verifiziert Claims NUR gegen autoritative Quellen.
    
    Quellen-Hierarchie:
    1. Authority Files (GND, VIAF, LOC) - 95% Zuverlässigkeit
    2. Fact-Check APIs (Google, ClaimBuster) - 85% Zuverlässigkeit
    3. Academic Sources (CrossRef, Open Library) - 80% Zuverlässigkeit
    
    KEINE Wikipedia/Wikidata!
    """
    
    def __init__(self, use_cache: bool = True, google_api_key: Optional[str] = None):
        self.authority_client = AuthorityFilesClient()
        self.factcheck_client = FactCheckClient(google_api_key=google_api_key)
        self.academic_client = AcademicSourcesClient()
        
        self.use_cache = use_cache
        self.cache = VerificationCache() if use_cache else None
    
    async def verify(
        self,
        claim: str,
        check_authority: bool = True,
        check_factcheck: bool = True,
        check_academic: bool = True,
        skip_cache: bool = False,
    ) -> AuthoritativeVerificationResult:
        """
        Verifiziert einen Claim gegen autoritative Quellen.
        
        Args:
            claim: Der zu verifizierende Claim
            check_authority: Authority Files prüfen (Tier 1)
            check_factcheck: Fact-Check APIs prüfen (Tier 2)
            check_academic: Akademische Quellen prüfen (Tier 3)
            skip_cache: Cache überspringen
            
        Returns:
            AuthoritativeVerificationResult
        """
        # Cache Check
        if self.cache and not skip_cache:
            cached = self.cache.get(claim)
            if cached:
                logger.info(f"Cache hit for: {claim[:50]}...")
                return cached
        
        results: List[AuthoritativeSourceResult] = []
        tasks = []
        
        # Tier 1: Authority Files
        if check_authority:
            tasks.extend([
                self.authority_client.search_gnd(claim),
                self.authority_client.search_viaf(claim),
                self.authority_client.search_loc(claim),
            ])
        
        # Tier 2: Fact-Check APIs
        if check_factcheck:
            tasks.extend([
                self.factcheck_client.search_google_factcheck(claim),
                self.factcheck_client.check_claimbuster(claim),
            ])
        
        # Tier 3: Academic Sources
        if check_academic:
            tasks.extend([
                self.academic_client.search_crossref(claim),
                self.academic_client.search_open_library(claim),
            ])
        
        # Parallel ausführen
        try:
            source_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in source_results:
                if isinstance(result, AuthoritativeSourceResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Source query failed: {result}")
                    
        except Exception as e:
            logger.error(f"Verification failed: {e}")
        
        # Ergebnis aggregieren
        verification = self._aggregate_results(claim, results)
        
        # Cache speichern
        if self.cache and not skip_cache:
            self.cache.set(claim, verification)
        
        return verification
    
    def _aggregate_results(
        self, 
        claim: str, 
        results: List[AuthoritativeSourceResult]
    ) -> AuthoritativeVerificationResult:
        """Aggregiert Ergebnisse mit Tier-gewichteter Bewertung."""
        
        found_results = [r for r in results if r.found]
        
        if not found_results:
            return AuthoritativeVerificationResult(
                query=claim,
                verified=False,
                confidence=0.0,
                sources_checked=len(results),
                sources_found=0,
                results=results,
                summary="No authoritative sources found for this claim.",
                recommendation="This claim could not be verified against authoritative sources. Additional research recommended.",
            )
        
        # Höchste Tier finden
        tier_order = [SourceTier.TIER_1_AUTHORITY, SourceTier.TIER_2_FACTCHECK, SourceTier.TIER_3_ACADEMIC, SourceTier.TIER_4_ENCYCLOPEDIA]
        highest_tier = None
        for tier in tier_order:
            if any(r.tier == tier for r in found_results):
                highest_tier = tier
                break
        
        # Tier-gewichteten Score berechnen
        total_weight = sum(r.reliability_score for r in found_results)
        weighted_relevance = sum(
            r.relevance_score * r.reliability_score 
            for r in found_results
        ) / total_weight if total_weight > 0 else 0
        
        # Confidence mit Tier-Bonus
        tier_bonus = {
            SourceTier.TIER_1_AUTHORITY: 0.15,
            SourceTier.TIER_2_FACTCHECK: 0.10,
            SourceTier.TIER_3_ACADEMIC: 0.05,
            SourceTier.TIER_4_ENCYCLOPEDIA: 0.0,
        }
        
        confidence = min(1.0, weighted_relevance + tier_bonus.get(highest_tier, 0))
        
        # Fact-Check Rating extrahieren
        fact_check_rating = None
        fact_check_results = [r for r in found_results if r.source_type == "fact_check" and r.rating]
        if fact_check_results:
            fact_check_rating = fact_check_results[0].rating
        
        # Summary erstellen
        tier_names = {
            SourceTier.TIER_1_AUTHORITY: "authority files (highest reliability)",
            SourceTier.TIER_2_FACTCHECK: "fact-checking organizations",
            SourceTier.TIER_3_ACADEMIC: "academic sources",
        }
        
        summary_parts = [f"Found in {len(found_results)} authoritative sources."]
        
        if highest_tier:
            summary_parts.append(f"Highest tier: {tier_names.get(highest_tier, highest_tier.value)}.")
        
        if fact_check_rating:
            summary_parts.append(f"Fact-check rating: {fact_check_rating}.")
        
        # Recommendation
        if confidence >= 0.8:
            recommendation = "High confidence from authoritative sources. This claim is well-documented."
        elif confidence >= 0.6:
            recommendation = "Moderate support from authoritative sources. Some verification achieved."
        elif confidence >= 0.4:
            recommendation = "Limited authoritative support. Additional verification recommended."
        else:
            recommendation = "Minimal authoritative support found. Treat this claim with caution."
        
        return AuthoritativeVerificationResult(
            query=claim,
            verified=confidence > 0.5,
            confidence=round(confidence, 3),
            sources_checked=len(results),
            sources_found=len(found_results),
            highest_tier_found=highest_tier,
            results=results,
            fact_check_rating=fact_check_rating,
            summary=" ".join(summary_parts),
            recommendation=recommendation,
        )
    
    async def quick_verify(self, claim: str) -> Dict[str, Any]:
        """Schnelle Verifikation für UI."""
        result = await self.verify(claim)
        
        return {
            "verified": result.verified,
            "confidence": result.confidence,
            "sources_found": result.sources_found,
            "highest_tier": result.highest_tier_found.value if result.highest_tier_found else None,
            "fact_check_rating": result.fact_check_rating,
            "summary": result.summary,
        }
    
    async def close(self):
        """Schließt alle Clients."""
        await self.authority_client.close()
        await self.factcheck_client.close()
        await self.academic_client.close()


# =============================================================================
# Singleton
# =============================================================================

_verifier_instance: Optional[AuthoritativeSourceVerifier] = None


def get_authoritative_verifier() -> AuthoritativeSourceVerifier:
    """Gibt Authoritative Source Verifier Instanz zurück."""
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = AuthoritativeSourceVerifier()
    return _verifier_instance