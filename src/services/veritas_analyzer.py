"""
Veritas Analyzer v2 - Mit autoritativen Quellen und optimiertem LLM.

Features:
- Autoritative Quellen (GND, VIAF, Wikidata, LOC)
- Few-Shot Prompting mit historischen Beispielen
- Chain-of-Thought Reasoning
- RAG mit Mythen-Datenbank
- Quellenverifizierung
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional
from uuid import uuid4

from src.config import get_settings
from src.llm.client import get_llm_client
from src.data.myths_database import get_myths_database
from src.services.veritas_prompts import (
    HISTORIAN_SYSTEM_PROMPT,
    FACT_CHECKER_SYSTEM_PROMPT,
    build_fact_check_prompt,
    build_context_analysis_prompt,
    build_narrative_analysis_prompt,
    build_claim_extraction_prompt,
    build_verdict_prompt,
)
from src.models.veritas_schema import (
    FactStatus,
    ContextStatus,
    NarrativeStatus,
    ConfidenceLevel,
    Source,
    SourceType,
    ClaimAnalysis,
    ContextAnalysis,
    NarrativeMatch,
    FullAnalysis,
    HistoricalMyth,
)

logger = logging.getLogger(__name__)


def _get_enum_value(enum_or_str) -> str:
    """Safely get string value from enum or string."""
    if isinstance(enum_or_str, str):
        return enum_or_str
    return enum_or_str.value if hasattr(enum_or_str, "value") else str(enum_or_str)


# Optional: Authority Client (nur wenn verfuegbar)
try:
    from src.datasources.authority_client import (
        AuthorityClient,
        AuthorityType,
        get_authority_client,
    )

    AUTHORITY_AVAILABLE = True
except ImportError:
    AUTHORITY_AVAILABLE = False
    logger.warning("Authority client not available")


class VeritasAnalyzer:
    """
    Veritas Analyzer v2 mit autoritativen Quellen.

    Workflow:
    1. Mythen-DB Check (schnell, offline)
    2. Authority Sources Check (GND, VIAF, Wikidata)
    3. LLM-Analyse mit Few-Shot und RAG
    4. Verdict-Generierung
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        use_authority_sources: bool = True,
    ):
        settings = get_settings()

        self.provider = provider or settings.llm_provider
        self.model = model or settings.get_active_model()
        self.use_authority_sources = use_authority_sources and AUTHORITY_AVAILABLE

        # LLM Client
        self.client = get_llm_client(provider=self.provider, model=self.model)

        # Mythen-Datenbank
        self.myths_db = get_myths_database()

        # Authority Client
        self.authority_client = None
        if self.use_authority_sources:
            self.authority_client = get_authority_client()

        logger.info(f"VeritasAnalyzer v2 initialisiert: {self.provider} / {self.model}")
        logger.info(
            f"Authority Sources: {'aktiviert' if self.use_authority_sources else 'deaktiviert'}"
        )

    async def analyze(
        self,
        text: str,
        deep_analysis: bool = True,
        language: str = "auto",
        verify_with_sources: bool = True,
    ) -> FullAnalysis:
        """
        Vollstaendige Analyse mit autoritativen Quellen.
        """
        analysis_id = str(uuid4())[:8]
        logger.info(f"[{analysis_id}] Starte Analyse...")

        # Sprache erkennen
        if language == "auto":
            language = self._detect_language(text)

        # === PHASE 1: Mythen-DB Check ===
        myth_match = self._match_myth(text)

        if myth_match:
            logger.info(f"[{analysis_id}] Bekannter Mythos: {myth_match.id}")
            return self._create_analysis_from_myth(
                analysis_id, text, myth_match, language
            )

        # === PHASE 2: Claims extrahieren ===
        claims_data = await self._extract_claims(text)
        claims_list = claims_data.get("claims", [])

        if not claims_list:
            claims_list = [
                {"claim": text, "entities": [], "dates": [], "locations": []}
            ]

        # === PHASE 3: Jeden Claim analysieren ===
        analyzed_claims = []
        all_sources = []

        for claim_data in claims_list:
            claim_text = claim_data.get("claim", text)
            entities = claim_data.get("entities", [])

            # Nochmal Mythen-Check
            claim_myth = self._match_myth(claim_text)

            if claim_myth:
                claim_analysis = self._create_claim_from_myth(
                    claim_myth, claim_text, language
                )
                all_sources.extend(claim_myth.sources)
                analyzed_claims.append(claim_analysis)
                continue

            # Authority Sources abfragen
            source_info = "Keine externen Quellen verfuegbar."
            authority_sources = []

            if verify_with_sources and self.authority_client and entities:
                source_info, authority_sources = await self._gather_authority_info(
                    entities
                )
                all_sources.extend(authority_sources)

            # Mythen-Kontext fuer RAG
            myths_context = self._get_relevant_myths_context(claim_text)

            # LLM Fakten-Check mit Few-Shot
            fact_result = await self._check_facts_with_sources(
                claim_text, source_info, myths_context
            )

            # Kontext-Analyse (optional)
            context_result = None
            if deep_analysis:
                context_result = await self._analyze_context(claim_text, fact_result)

            # Narrativ-Analyse (optional)
            narrative_result = None
            if deep_analysis:
                narrative_result = await self._analyze_narrative(claim_text)

            # ClaimAnalysis erstellen
            claim_analysis = ClaimAnalysis(
                original_claim=claim_text,
                language=language,
                entities=entities,
                dates=claim_data.get("dates", []),
                locations=claim_data.get("locations", []),
                fact_status=FactStatus(fact_result.get("fact_status", "unverified")),
                context_status=(
                    ContextStatus(context_result.get("context_status", "complete"))
                    if context_result
                    else ContextStatus.COMPLETE
                ),
                narrative_status=(
                    NarrativeStatus(narrative_result.get("narrative_status", "neutral"))
                    if narrative_result
                    else NarrativeStatus.NEUTRAL
                ),
                confidence=self._score_to_level(
                    fact_result.get("confidence_score", 0.5)
                ),
                confidence_score=fact_result.get("confidence_score", 0.5),
                what_is_true=fact_result.get("what_is_true", []),
                what_is_false=fact_result.get("what_is_false", []),
                what_is_missing=(
                    context_result.get("important_omissions", [])
                    if context_result
                    else []
                ),
                narrative_match=(
                    self._build_narrative_match(narrative_result)
                    if narrative_result
                    else None
                ),
                sources_used=authority_sources,
            )

            analyzed_claims.append(claim_analysis)

        # === PHASE 4: Gesamturteil ===
        verdict_result = await self._generate_verdict(text, analyzed_claims, None, None)

        return FullAnalysis(
            id=analysis_id,
            timestamp=datetime.utcnow(),
            language=language,
            input_text=text,
            input_type="claim" if len(text) < 500 else "article",
            claims_found=len(analyzed_claims),
            claims=analyzed_claims,
            context_analysis=None,
            overall_verdict=verdict_result.get("verdict", "unverifiable"),
            verdict_explanation=verdict_result.get("verdict_explanation", ""),
            summary_for_users=verdict_result.get("summary_for_users", ""),
            all_sources=all_sources,
            recommendation=verdict_result.get("recommendation"),
        )

    async def quick_check(self, claim: str) -> dict:
        """Schneller Check gegen Mythen-DB."""
        myth = self._match_myth(claim)

        if myth:
            return {
                "found": True,
                "myth_id": myth.id,
                "claim": myth.claim,
                "status": _get_enum_value(myth.status),
                "truth": myth.truth,
                "origin": myth.origin.source,
                "sources": [s.title for s in myth.sources[:3]],
            }

        return {"found": False, "message": "No known myth found."}

    async def verify_against_authorities(
        self, person_name: str, claim_type: str, claimed_value: str
    ) -> dict:
        """Verifiziert Fakt gegen autoritative Quellen."""
        if not self.authority_client:
            return {"error": "Authority sources not enabled"}

        return await self.authority_client.verify_person_fact(
            person_name, claim_type, claimed_value
        )

    # =========================================================================
    # Private: Authority Sources
    # =========================================================================

    async def _gather_authority_info(
        self, entities: list[str]
    ) -> tuple[str, list[Source]]:
        """Sammelt Informationen aus autoritativen Quellen."""

        info_parts = []
        sources = []

        for entity in entities[:3]:  # Max 3 Entities
            try:
                # Parallel in mehreren Quellen suchen
                results = await self.authority_client.search_all(
                    entity, sources=["gnd", "wikidata"], limit_per_source=1
                )

                # GND Ergebnisse
                for record in results.get("gnd", []):
                    info_parts.append(f"[GND] {record.label}")
                    if record.description:
                        info_parts.append(f"  - {record.description}")
                    if record.birth_date:
                        info_parts.append(f"  - Geboren: {record.birth_date}")
                    if record.death_date:
                        info_parts.append(f"  - Gestorben: {record.death_date}")

                    sources.append(
                        Source(
                            type=SourceType.AUTHORITY,
                            title=f"GND: {record.label}",
                            url=record.uri,
                            reliability=ConfidenceLevel.HIGH,
                        )
                    )

                # Wikidata Fakten
                for record in results.get("wikidata", []):
                    if record.id:
                        facts = await self.authority_client.get_wikidata_facts(
                            record.id
                        )
                        for fact in facts[:5]:
                            info_parts.append(f"[Wikidata] {fact.claim}: {fact.value}")

                        if facts:
                            sources.append(
                                Source(
                                    type=SourceType.AUTHORITY,
                                    title=f"Wikidata: {record.label}",
                                    url=record.uri,
                                    reliability=ConfidenceLevel.HIGH,
                                )
                            )

            except Exception as e:
                logger.warning(f"Authority lookup failed for '{entity}': {e}")

        return (
            "\n".join(info_parts)
            if info_parts
            else "Keine autoritativen Daten gefunden."
        ), sources

    # =========================================================================
    # Private: LLM Analysis with Few-Shot
    # =========================================================================

    async def _extract_claims(self, text: str) -> dict:
        """Extrahiert Claims mit optimiertem Prompt."""
        prompt = build_claim_extraction_prompt(text)

        try:
            result = await self.client.extract_json(
                prompt=prompt, system=HISTORIAN_SYSTEM_PROMPT
            )
            return result
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return {"claims": [], "language": "de"}

    async def _check_facts_with_sources(
        self, claim: str, source_info: str, myths_context: str
    ) -> dict:
        """Faktencheck mit Few-Shot und Quelleninformation."""

        prompt = build_fact_check_prompt(claim, source_info, myths_context)

        try:
            result = await self.client.extract_json(
                prompt=prompt, system=FACT_CHECKER_SYSTEM_PROMPT
            )

            # Reasoning loggen
            if "reasoning" in result:
                logger.info(f"LLM Reasoning: {result['reasoning']}")

            return result

        except Exception as e:
            logger.error(f"Fact check failed: {e}")
            return {
                "fact_status": "unverified",
                "confidence_score": 0.3,
                "what_is_true": [],
                "what_is_false": [],
            }

    async def _analyze_context(self, claim: str, fact_result: dict) -> dict:
        """Kontext-Analyse mit Few-Shot."""
        prompt = build_context_analysis_prompt(
            claim, fact_result.get("fact_status", "unknown")
        )

        try:
            return await self.client.extract_json(
                prompt=prompt, system=HISTORIAN_SYSTEM_PROMPT
            )
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return {"context_status": "complete"}

    async def _analyze_narrative(self, claim: str) -> dict:
        """Narrativ-Analyse mit Few-Shot."""

        # Bekannte Narrative als Kontext
        narratives_info = "\n".join(
            [
                f"- {n.id}: {n.name} - {n.description[:100]}..."
                for n in self.myths_db.narratives.values()
            ]
        )

        prompt = build_narrative_analysis_prompt(claim, narratives_info)

        try:
            return await self.client.extract_json(
                prompt=prompt, system=HISTORIAN_SYSTEM_PROMPT
            )
        except Exception as e:
            logger.error(f"Narrative analysis failed: {e}")
            return {"narrative_status": "neutral"}

    async def _generate_verdict(
        self,
        original_text: str,
        claims: list[ClaimAnalysis],
        context_result: Optional[dict],
        narrative_result: Optional[dict],
    ) -> dict:
        """Generiert Gesamturteil."""

        # Claims zusammenfassen
        claims_summary = "\n".join(
            [
                f"- {c.original_claim[:80]}: {_get_enum_value(c.fact_status)} ({c.confidence_score:.0%})"
                for c in claims
            ]
        )

        prompt = build_verdict_prompt(
            original_text[:500],
            claims_summary,
            str(context_result) if context_result else "Nicht analysiert",
            str(narrative_result) if narrative_result else "Nicht analysiert",
        )

        try:
            return await self.client.extract_json(
                prompt=prompt, system=HISTORIAN_SYSTEM_PROMPT
            )
        except Exception as e:
            logger.error(f"Verdict generation failed: {e}")
            return self._fallback_verdict(claims)

    def _fallback_verdict(self, claims: list[ClaimAnalysis]) -> dict:
        """Fallback wenn LLM-Verdict fehlschlaegt."""
        if not claims:
            return {
                "verdict": "unverifiable",
                "verdict_explanation": "No claims analyzed.",
                "summary_for_users": "Analysis not possible.",
            }

        statuses = [_get_enum_value(c.fact_status) for c in claims]

        if "false" in statuses or "myth" in statuses:
            return {
                "verdict": "false",
                "verdict_explanation": "At least one claim is false.",
                "summary_for_users": "The analysis identified false claims.",
            }
        elif all(s == "confirmed" for s in statuses):
            return {
                "verdict": "historically_accurate",
                "verdict_explanation": "All claims confirmed.",
                "summary_for_users": "The statements are historically correct.",
            }
        else:
            return {
                "verdict": "unverifiable",
                "verdict_explanation": "Not all statements could be verified.",
                "summary_for_users": "Some statements could not be verified.",
            }

    # =========================================================================
    # Private: Helpers
    # =========================================================================

    def _detect_language(self, text: str) -> str:
        """Erkennt Sprache."""
        german_indicators = [
            "der",
            "die",
            "das",
            "und",
            "ist",
            "war",
            "ä",
            "ö",
            "ü",
            "ß",
        ]
        text_lower = text.lower()
        german_count = sum(1 for ind in german_indicators if ind in text_lower)
        return "de" if german_count >= 2 else "en"

    def _match_myth(self, claim: str) -> Optional[HistoricalMyth]:
        """Prueft ob Claim einem Mythos entspricht."""
        claim_lower = claim.lower().strip()

        best_match = None
        best_score = 0

        for myth in self.myths_db.myths.values():
            score = 0

            # Direct claim match (highest priority)
            if myth.claim.lower() in claim_lower or claim_lower in myth.claim.lower():
                score += 10

            if myth.claim_en and (
                myth.claim_en.lower() in claim_lower
                or claim_lower in myth.claim_en.lower()
            ):
                score += 10

            # Keyword matches
            keyword_matches = sum(
                1 for kw in myth.keywords if kw.lower() in claim_lower
            )
            score += keyword_matches * 2

            # Text similarity
            similarity = self._text_similarity(claim_lower, myth.claim.lower())
            score += similarity * 5

            if myth.claim_en:
                similarity_en = self._text_similarity(
                    claim_lower, myth.claim_en.lower()
                )
                score += similarity_en * 5

            if score > best_score:
                best_score = score
                best_match = myth

        # Lower threshold for better matching
        return best_match if best_score >= 2 else None

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Jaccard similarity."""
        words1 = set(re.findall(r"\w+", text1.lower()))
        words2 = set(re.findall(r"\w+", text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _get_relevant_myths_context(self, claim: str) -> str:
        """Findet relevante Mythen fuer RAG."""
        related = self.myths_db.search_myths(claim)[:3]

        if not related:
            return ""

        context_parts = ["RELEVANTE MYTHEN AUS DATENBANK:"]
        for myth in related:
            context_parts.append(f"- {myth.claim}")
            context_parts.append(f"  Status: {_get_enum_value(myth.status)}")
            context_parts.append(f"  Wahrheit: {myth.truth[:150]}...")

        return "\n".join(context_parts)

    def _create_analysis_from_myth(
        self, analysis_id: str, text: str, myth: HistoricalMyth, language: str
    ) -> FullAnalysis:
        """Erstellt Analyse aus Mythen-DB."""

        claim_analysis = self._create_claim_from_myth(myth, text, language)

        status_str = _get_enum_value(myth.status)
        if status_str in ["false", "myth"]:
            verdict = "false"
            explanation = "This claim is a known historical myth."
        else:
            verdict = status_str
            explanation = "See detailed analysis."

        summary = myth.truth if language == "de" else (myth.truth_en or myth.truth)

        return FullAnalysis(
            id=analysis_id,
            timestamp=datetime.utcnow(),
            language=language,
            input_text=text,
            input_type="claim",
            claims_found=1,
            claims=[claim_analysis],
            context_analysis=None,
            overall_verdict=verdict,
            verdict_explanation=explanation,
            summary_for_users=summary,
            all_sources=myth.sources,
            recommendation=f"Ursprung: {myth.origin.source} ({myth.origin.date or 'unbekannt'})",
        )

    def _create_claim_from_myth(
        self, myth: HistoricalMyth, original_claim: str, language: str
    ) -> ClaimAnalysis:
        """Erstellt ClaimAnalysis aus Mythos."""

        status_str = _get_enum_value(myth.status)
        if status_str in ["false", "myth"]:
            what_is_true = [
                myth.truth if language == "de" else (myth.truth_en or myth.truth)
            ]
            what_is_false = [
                myth.claim if language == "de" else (myth.claim_en or myth.claim)
            ]
        else:
            what_is_true = [myth.truth]
            what_is_false = []

        return ClaimAnalysis(
            original_claim=original_claim,
            language=language,
            entities=[],
            dates=[],
            locations=[],
            fact_status=myth.status,
            context_status=ContextStatus.COMPLETE,
            narrative_status=NarrativeStatus.NEUTRAL,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.95,
            what_is_true=what_is_true,
            what_is_false=what_is_false,
            what_is_missing=[],
            related_myths=[myth.id],
            sources_used=myth.sources,
        )

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """Konvertiert Score zu Level."""
        if score >= 0.9:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCLEAR

    def _build_narrative_match(
        self, narrative_result: dict
    ) -> Optional[NarrativeMatch]:
        """Erstellt NarrativeMatch."""
        narrative_id = narrative_result.get("matched_narrative_id")

        pattern = None
        if narrative_id:
            pattern = self.myths_db.narratives.get(narrative_id)

        return NarrativeMatch(
            matched_pattern=pattern,
            confidence=narrative_result.get("matching_confidence", 0.0),
            matching_elements=narrative_result.get("matching_elements", []),
            is_known_propaganda=narrative_result.get("narrative_status")
            == "propaganda",
            origin_tracking=narrative_result.get("origin_hint"),
        )

    def search_myths(self, query: str) -> list[dict]:
        """Sucht in Mythen-DB."""
        myths = self.myths_db.search_myths(query)
        return [
            {
                "id": m.id,
                "claim": m.claim,
                "status": _get_enum_value(m.status),
                "truth": m.truth[:200] + "..." if len(m.truth) > 200 else m.truth,
                "category": _get_enum_value(m.category),
            }
            for m in myths
        ]

    def get_myth(self, myth_id: str) -> Optional[HistoricalMyth]:
        """Gibt Mythos nach ID zurueck."""
        return self.myths_db.get_myth(myth_id)


# =============================================================================
# Singleton
# =============================================================================

_analyzer_instance: Optional[VeritasAnalyzer] = None


def get_analyzer() -> VeritasAnalyzer:
    """Gibt Analyzer-Instanz zurueck."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = VeritasAnalyzer()
    return _analyzer_instance
