"""
Veritas LLM Prompts - Optimiert fuer historische Faktenanalyse.

Techniken:
- Few-Shot Learning mit historischen Beispielen
- Chain-of-Thought Reasoning
- Structured Output (JSON Schema)
- Anti-Hallucination Instructions
- RAG-Integration mit Mythen-DB
"""

from typing import Optional
from src.models.veritas_schema import FactStatus, ContextStatus, NarrativeStatus


# =============================================================================
# System Prompts
# =============================================================================

HISTORIAN_SYSTEM_PROMPT = """Du bist ein kritischer Historiker mit Expertise in:
- Quellenkritik und historiographische Methoden
- Erkennung von Propaganda und Geschichtsmythen
- Faktenverifizierung gegen Primaerquellen

WICHTIGE REGELN:
1. NIEMALS Fakten erfinden - wenn du unsicher bist, sage es
2. Unterscheide klar zwischen Fakten, Interpretationen und Mythen
3. Beruecksichtige den historischen Kontext
4. Erkenne Propaganda-Muster und Narrativ-Verzerrungen
5. Gib Konfidenzwerte NUR basierend auf Quellenqualitaet

Du antwortest IMMER im angeforderten JSON-Format."""

FACT_CHECKER_SYSTEM_PROMPT = """Du bist ein Faktenpruefer fuer historische Behauptungen.

DEINE AUFGABE:
1. Analysiere die Behauptung Schritt fuer Schritt
2. Identifiziere pruefbare Fakten
3. Vergleiche mit den bereitgestellten Quelleninformationen
4. Bewerte die Korrektheit STRENG basierend auf Quellen

ANTI-HALLUCINATION:
- Erfinde KEINE Informationen
- Wenn keine Quelle vorhanden: status = "unverified"
- Wenn Quelle widerspricht: status = "false"
- Nur bei klarer Quellenbestaetigung: status = "confirmed"

Antworte NUR im JSON-Format."""


# =============================================================================
# Few-Shot Examples
# =============================================================================

FEW_SHOT_FACT_CHECK = """
BEISPIEL 1:
Behauptung: "Napoleon war klein"
Quelleninformation: "Napoleon war 1,69m gross. Durchschnittliche Groesse franzoesischer Maenner 1800: 1,64m"
Analyse:
{
    "reasoning": [
        "1. Behauptung: Napoleon war klein",
        "2. Quelle sagt: 1,69m Koerpergroesse",
        "3. Durchschnitt damals: 1,64m",
        "4. 1,69m > 1,64m = ueberdurchschnittlich",
        "5. 'Klein' ist daher faktisch falsch"
    ],
    "fact_status": "myth",
    "confidence_score": 0.95,
    "what_is_true": ["Napoleon war 1,69m gross", "Dies war ueberdurchschnittlich fuer seine Zeit"],
    "what_is_false": ["Napoleon war klein"],
    "explanation": "Der Mythos entstand durch britische Kriegspropaganda und Verwechslung der Masseinheiten."
}

BEISPIEL 2:
Behauptung: "Einstein fiel in Mathematik durch"
Quelleninformation: "Einsteins Maturitaetszeugnis 1896 zeigt Note 6 in Mathematik. Schweizer Notensystem: 6 = beste Note."
Analyse:
{
    "reasoning": [
        "1. Behauptung: Einstein fiel durch",
        "2. Quelle: Zeugnis zeigt Note 6",
        "3. Schweizer System: 6 = beste Note (nicht schlechteste!)",
        "4. Einstein hatte Bestnote, nicht Durchfall"
    ],
    "fact_status": "false",
    "confidence_score": 0.98,
    "what_is_true": ["Einstein hatte Note 6 im Zeugnis"],
    "what_is_false": ["Einstein fiel durch", "Einstein war schlecht in Mathematik"],
    "explanation": "Missverstaendnis durch unterschiedliche Notensysteme (CH: 6=best, DE: 6=schlechteste)"
}

BEISPIEL 3:
Behauptung: "Der Reichstag wurde 1933 von den Nazis angezuendet"
Quelleninformation: "Marinus van der Lubbe wurde fuer den Reichstagsbrand verurteilt. Historiker debattieren noch ueber moegliche NS-Beteiligung."
Analyse:
{
    "reasoning": [
        "1. Behauptung: Nazis zuendeten Reichstag an",
        "2. Verurteilter Taeter: van der Lubbe (Einzeltaeter?)",
        "3. NS-Beteiligung: historisch umstritten",
        "4. Keine definitive Quellenbestaetigung fuer NS-Urheberschaft"
    ],
    "fact_status": "disputed",
    "confidence_score": 0.6,
    "what_is_true": ["Der Reichstag brannte am 27.2.1933", "Van der Lubbe wurde verurteilt"],
    "what_is_false": [],
    "explanation": "Die genaue Urheberschaft ist unter Historikern umstritten. Gaengige Theorien: Einzeltaeter, NS-Komplott, oder Kombination."
}
"""

FEW_SHOT_CONTEXT_ANALYSIS = """
BEISPIEL 1:
Behauptung: "Die Alliierten bombardierten Dresden"
Analyse:
{
    "context_status": "simplified",
    "missing_timeframe": "Kontext des Gesamtkrieges 1939-1945 fehlt",
    "missing_perspectives": ["Deutsche Luftangriffe auf London, Coventry, Rotterdam", "Militaerische Ziele in Dresden"],
    "missing_causes": ["Totaler Krieg seit 1943", "Strategische Bombardierung als Doktrin beider Seiten"],
    "missing_consequences": ["Kriegsende 3 Monate spaeter", "Nachkriegspropaganda"],
    "important_omissions": ["Dresden war Verkehrsknotenpunkt", "Ruestungsindustrie vorhanden"]
}

BEISPIEL 2:
Behauptung: "Israel wurde 1948 gegruendet"
Analyse:
{
    "context_status": "complete",
    "missing_timeframe": null,
    "missing_perspectives": [],
    "missing_causes": [],
    "missing_consequences": [],
    "important_omissions": []
}
(Dies ist eine neutrale Faktenaussage ohne Wertung)
"""

FEW_SHOT_NARRATIVE = """
BEISPIEL 1:
Behauptung: "Die Wehrmacht wusste nichts von den KZs"
Bekanntes Narrativ: "Saubere Wehrmacht" - Entlastungsnarrativ
Analyse:
{
    "narrative_status": "propaganda",
    "matched_narrative_id": "clean_army",
    "matching_confidence": 0.92,
    "matching_elements": [
        "Behauptung der Unwissenheit",
        "Trennung von Wehrmacht und SS",
        "Entlastung der regulaeren Soldaten"
    ],
    "likely_purpose": "Rehabilitation von Veteranen, Vermeidung von Kollektivschuld",
    "origin_hint": "Nuernberger Verteidigung 1945-46, Veteranenverbaende Nachkriegszeit"
}

BEISPIEL 2:
Behauptung: "Napoleon reformierte das Rechtssystem"
Analyse:
{
    "narrative_status": "neutral",
    "matched_narrative_id": null,
    "matching_confidence": 0.0,
    "matching_elements": [],
    "likely_purpose": null,
    "origin_hint": null
}
(Dies ist eine faktische Aussage ohne erkennbare Narrativ-Verzerrung)
"""


# =============================================================================
# Main Prompts with Few-Shot
# =============================================================================


def build_fact_check_prompt(
    claim: str, source_info: str, myths_context: Optional[str] = None
) -> str:
    """Baut den Faktencheck-Prompt mit Few-Shot und Kontext."""

    context_section = ""
    if myths_context:
        context_section = f"""
BEKANNTE MYTHEN AUS DATENBANK:
{myths_context}
"""

    return f"""{FEW_SHOT_FACT_CHECK}

---

JETZT DEINE AUFGABE:

Behauptung: "{claim}"

Quelleninformation:
{source_info}
{context_section}

Analysiere diese Behauptung Schritt fuer Schritt wie in den Beispielen.
Antworte NUR im JSON-Format:
{{
    "reasoning": ["Schritt 1...", "Schritt 2...", ...],
    "fact_status": "confirmed|likely|disputed|unverified|false|myth",
    "confidence_score": 0.0-1.0,
    "what_is_true": ["..."],
    "what_is_false": ["..."],
    "explanation": "..."
}}"""


def build_context_analysis_prompt(claim: str, fact_status: str) -> str:
    """Baut den Kontext-Analyse-Prompt."""

    return f"""{FEW_SHOT_CONTEXT_ANALYSIS}

---

JETZT DEINE AUFGABE:

Behauptung: "{claim}"
Fakten-Status: {fact_status}

Pruefe, welcher wichtige Kontext fehlt.
Antworte NUR im JSON-Format:
{{
    "context_status": "complete|simplified|selective|decontextualized|misleading",
    "missing_timeframe": "..." oder null,
    "missing_perspectives": ["..."],
    "missing_causes": ["..."],
    "missing_consequences": ["..."],
    "important_omissions": ["..."]
}}"""


def build_narrative_analysis_prompt(claim: str, known_narratives: str) -> str:
    """Baut den Narrativ-Analyse-Prompt."""

    return f"""{FEW_SHOT_NARRATIVE}

---

BEKANNTE NARRATIVE IN DATENBANK:
{known_narratives}

---

JETZT DEINE AUFGABE:

Behauptung: "{claim}"

Pruefe, ob diese Behauptung einem bekannten Narrativ oder Propaganda-Muster entspricht.
Antworte NUR im JSON-Format:
{{
    "narrative_status": "neutral|perspectival|biased|propaganda|revisionism",
    "matched_narrative_id": "ID" oder null,
    "matching_confidence": 0.0-1.0,
    "matching_elements": ["..."],
    "likely_purpose": "..." oder null,
    "origin_hint": "..." oder null
}}"""


def build_claim_extraction_prompt(text: str) -> str:
    """Baut den Claim-Extraktions-Prompt."""

    return f"""Analysiere den folgenden Text und extrahiere alle historischen Behauptungen.

TEXT:
{text}

REGELN:
1. Jede ueberpruefbare Aussage ist eine Behauptung
2. Trenne zusammengesetzte Aussagen in einzelne Claims
3. Identifiziere Entitaeten (Personen, Orte, Daten)

Antworte NUR im JSON-Format:
{{
    "claims": [
        {{
            "claim": "Die Behauptung als klarer Aussagesatz",
            "entities": ["Person1", "Ort1"],
            "dates": ["1945"],
            "locations": ["Berlin"],
            "implicit_assumptions": ["Falls vorhanden"]
        }}
    ],
    "language": "de" oder "en",
    "main_topic": "Hauptthema"
}}"""


def build_verdict_prompt(
    original_text: str,
    claims_summary: str,
    context_summary: str,
    narrative_summary: str,
) -> str:
    """Baut den Gesamturteil-Prompt."""

    return f"""Basierend auf der vollstaendigen Analyse, erstelle ein Gesamturteil.

ORIGINAL-TEXT:
{original_text}

FAKTEN-ANALYSE:
{claims_summary}

KONTEXT-ANALYSE:
{context_summary}

NARRATIV-ANALYSE:
{narrative_summary}

BEWERTUNGS-SCHEMA:
- historically_accurate: Fakten und Kontext stimmen mit historischem Konsens ueberein
- simplified: Grundsaetzlich korrekt, aber wichtige Nuancen fehlen
- selective: Nur ausgewaehlte Fakten, wichtige Aspekte fehlen absichtlich
- decontextualized: Fakten aus dem Zusammenhang gerissen
- reinterpreted: Fakten stimmen, aber Interpretation ist fragwuerdig
- propaganda: Folgt bekannten manipulativen Narrativ-Mustern
- false: Kernbehauptungen sind faktisch falsch
- unverifiable: Keine ausreichenden Quellen zur Pruefung

Erstelle ein faires, sachliches Urteil.
Antworte NUR im JSON-Format:
{{
    "verdict": "historically_accurate|simplified|selective|decontextualized|reinterpreted|propaganda|false|unverifiable",
    "verdict_explanation": "Begruendung des Urteils",
    "summary_for_users": "Verstaendliche Zusammenfassung (2-3 Saetze, fuer Laien)",
    "recommendation": "Was sollte der Nutzer beachten oder weiter recherchieren?"
}}"""


# =============================================================================
# Output Schemas (fuer Structured Output Validation)
# =============================================================================

FACT_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Schrittweise Analyse",
        },
        "fact_status": {
            "type": "string",
            "enum": ["confirmed", "likely", "disputed", "unverified", "false", "myth"],
        },
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "what_is_true": {"type": "array", "items": {"type": "string"}},
        "what_is_false": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
    "required": ["fact_status", "confidence_score", "what_is_true", "what_is_false"],
}

CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "context_status": {
            "type": "string",
            "enum": [
                "complete",
                "simplified",
                "selective",
                "decontextualized",
                "misleading",
            ],
        },
        "missing_timeframe": {"type": ["string", "null"]},
        "missing_perspectives": {"type": "array", "items": {"type": "string"}},
        "missing_causes": {"type": "array", "items": {"type": "string"}},
        "missing_consequences": {"type": "array", "items": {"type": "string"}},
        "important_omissions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["context_status"],
}

NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "narrative_status": {
            "type": "string",
            "enum": ["neutral", "perspectival", "biased", "propaganda", "revisionism"],
        },
        "matched_narrative_id": {"type": ["string", "null"]},
        "matching_confidence": {"type": "number"},
        "matching_elements": {"type": "array", "items": {"type": "string"}},
        "likely_purpose": {"type": ["string", "null"]},
        "origin_hint": {"type": ["string", "null"]},
    },
    "required": ["narrative_status"],
}
