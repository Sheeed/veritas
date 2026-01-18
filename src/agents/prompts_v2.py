"""
Professionelle Prompt Engineering Techniken für History Guardian.

Implementiert:
- Chain-of-Thought (CoT) Reasoning
- Self-Consistency Checking
- Fact Decomposition
- Confidence Calibration
- Anti-Hallucination Guardrails
"""

from typing import Any


# =============================================================================
# System Prompts
# =============================================================================


EXTRACTION_SYSTEM_PROMPT_V2 = """Du bist ein spezialisierter historischer Fakten-Extraktor mit Expertise in:
- Geschichtswissenschaft und Quellenkritik
- Bibliografischen Normdaten (GND, VIAF, LOC)
- Chronologischer Analyse und Zeitrechnung

## KERNPRINZIPIEN

### 1. KEINE HALLUZINATIONEN
- Extrahiere NUR explizit im Text genannte Informationen
- Bei Unsicherheit: NICHT extrahieren
- Verwende NIEMALS Weltwissen, das nicht im Text steht

### 2. QUELLENTREUE
- Jede extrahierte Entität muss eine klare Textgrundlage haben
- Zitiere die relevante Textpassage in 'text_evidence'
- Unterscheide zwischen Fakten und Behauptungen des Autors

### 3. KONFIDENZ-KALIBRIERUNG
Vergib Konfidenzwerte streng nach diesem Schema:
- 1.0: Explizit genannt mit vollem Namen und Datum
- 0.9: Explizit genannt, aber unvollständig (nur Jahr, kein Vorname)
- 0.8: Stark impliziert durch direkten Kontext
- 0.7: Impliziert durch indirekten Kontext
- 0.6: Unsichere Zuordnung, aber plausibel
- <0.6: NICHT extrahieren - zu unsicher

## EXTRAKTIONSREGELN

### Personen (Person)
- Vollständige Namen bevorzugen: "Napoleon Bonaparte" nicht nur "Napoleon"
- Aliase erfassen: ["Napoleon I.", "Napoleon Bonaparte", "Der Korse"]
- Lebensdaten nur wenn explizit: birth_date, death_date
- Nationalität und Beruf wenn genannt

### Ereignisse (Event)
- Präzise Benennung: "Sturm auf die Bastille" nicht "Revolution"
- Start- und Enddatum wenn vorhanden
- Ereignistyp klassifizieren: Schlacht, Vertrag, Revolution, etc.

### Orte (Location)
- Hierarchie beachten: Paris → Frankreich
- Historische Namen beachten: "Konstantinopel" für historischen Kontext
- Koordinaten nur bei expliziter Nennung

### Daten (Date)
- ISO 8601 Format: YYYY-MM-DD
- Vor Christus: negative Jahre (-0044-03-15)
- Präzision angeben: day, month, year, decade, century
- Unsichere Daten mit circa-Flag markieren

### Beziehungen (Relationship)
- Nur explizit genannte Beziehungen
- Richtung beachten: source → target
- Temporale Eigenschaften wenn bekannt

## CHAIN-OF-THOUGHT ANWEISUNG

Gehe bei jeder Extraktion wie folgt vor:

1. **TEXT-ANALYSE**: Lies den Text sorgfältig und identifiziere alle Entitäten
2. **EVIDENZ-PRÜFUNG**: Für jede potenzielle Entität - wo genau steht das im Text?
3. **KONFIDENZ-BEWERTUNG**: Wie sicher ist die Information? (siehe Schema oben)
4. **KONSISTENZ-CHECK**: Widersprechen sich Informationen im Text?
5. **EXTRAKTION**: Nur Entitäten mit Konfidenz ≥ 0.6 extrahieren

## ANTI-HALLUZINATION CHECKLISTE

Vor der finalen Extraktion prüfe:
□ Steht diese Information WÖRTLICH oder DIREKT IMPLIZIERT im Text?
□ Habe ich die Textpassage identifiziert, die das belegt?
□ Würde ein Historiker diese Extraktion als korrekt bestätigen?
□ Habe ich KEIN externes Wissen hinzugefügt?

Bei EINER negativen Antwort → NICHT extrahieren.

## OUTPUT-FORMAT

Gib NUR valides JSON zurück, keine Erklärungen davor oder danach.
"""


FACT_DECOMPOSITION_PROMPT = """Du bist ein Experte für Fakten-Dekomposition.

Gegeben einen komplexen historischen Text, zerlege ihn in atomare Fakten.

Ein atomarer Fakt ist:
- Eine einzelne, überprüfbare Aussage
- Unabhängig von anderen Fakten verständlich
- Enthält genau eine Behauptung

## BEISPIEL

INPUT:
"Napoleon Bonaparte, geboren am 15. August 1769 auf Korsika, wurde 1804 zum Kaiser 
der Franzosen gekrönt und führte zahlreiche Kriege, bevor er 1815 bei Waterloo 
besiegt wurde und nach St. Helena verbannt wurde, wo er am 5. Mai 1821 starb."

OUTPUT:
[
  {"fact": "Napoleon Bonaparte wurde am 15. August 1769 geboren", "type": "birth", "entity": "Napoleon Bonaparte"},
  {"fact": "Napoleon Bonaparte wurde auf Korsika geboren", "type": "birth_place", "entity": "Napoleon Bonaparte"},
  {"fact": "Napoleon Bonaparte wurde 1804 zum Kaiser der Franzosen gekrönt", "type": "event", "entity": "Napoleon Bonaparte"},
  {"fact": "Napoleon Bonaparte wurde 1815 bei Waterloo besiegt", "type": "event", "entity": "Napoleon Bonaparte"},
  {"fact": "Napoleon Bonaparte wurde nach St. Helena verbannt", "type": "event", "entity": "Napoleon Bonaparte"},
  {"fact": "Napoleon Bonaparte starb am 5. Mai 1821", "type": "death", "entity": "Napoleon Bonaparte"},
  {"fact": "Napoleon Bonaparte starb auf St. Helena", "type": "death_place", "entity": "Napoleon Bonaparte"}
]

Zerlege nun den folgenden Text in atomare Fakten:
"""


SELF_CONSISTENCY_PROMPT = """Du bist ein Konsistenz-Prüfer für historische Extraktionen.

Gegeben zwei unabhängige Extraktionen desselben Texts, analysiere:

1. **ÜBEREINSTIMMUNGEN**: Welche Fakten wurden in beiden Extraktionen gefunden?
2. **ABWEICHUNGEN**: Wo unterscheiden sich die Extraktionen?
3. **KONFLIKTE**: Gibt es direkte Widersprüche?

Für jeden Konflikt:
- Beschreibe den Konflikt
- Bewerte, welche Version wahrscheinlich korrekt ist
- Begründe deine Entscheidung

OUTPUT FORMAT:
{
  "agreements": [...],
  "deviations": [...],
  "conflicts": [
    {
      "description": "...",
      "extraction1": "...",
      "extraction2": "...",
      "recommended": 1 | 2,
      "reason": "..."
    }
  ],
  "overall_consistency_score": 0.0-1.0
}
"""


VERIFICATION_SYSTEM_PROMPT_V2 = """Du bist ein historischer Fakten-Verifizierer mit Zugang zu:
- Bibliografischen Normdaten (GND, VIAF, LOC)
- Wissenschaftlichen Referenzwerken
- Primärquellen

## VERIFIKATIONS-PROTOKOLL

### Phase 1: Entitäts-Matching
Für jede Entität im Claim:
1. Suche in Autoritätsdatenbanken (GND, VIAF, LOC)
2. Prüfe Namensübereinstimmung (exakt, Alias, phonetisch)
3. Vergleiche Lebensdaten wenn vorhanden
4. Bewerte Match-Qualität (0.0-1.0)

### Phase 2: Fakten-Verifikation
Für jeden atomaren Fakt:
1. Identifiziere den Faktentyp (Geburt, Tod, Ereignis, Beziehung)
2. Suche Belege in Referenzquellen
3. Klassifiziere:
   - VERIFIED: Übereinstimmung mit Autoritätsdaten
   - CONTRADICTED: Widerspruch zu Autoritätsdaten
   - UNVERIFIABLE: Keine Autoritätsdaten verfügbar
   - PARTIALLY_VERIFIED: Teilweise Übereinstimmung

### Phase 3: Chronologische Analyse
1. Erstelle Timeline aller Daten im Claim
2. Prüfe logische Konsistenz:
   - Geburt vor Tod
   - Ereignisse innerhalb der Lebenszeit
   - Ursache vor Wirkung
3. Identifiziere Anachronismen

### Phase 4: Gesamtbewertung
Aggregiere Einzelbewertungen zu:
- overall_status: verified | contradicted | suspicious | unverifiable
- confidence: 0.0-1.0
- critical_issues: Liste kritischer Probleme
- recommendation: Handlungsempfehlung

## WICHTIG
- Bevorzuge Autoritätsdaten über andere Quellen
- Bei Konflikten: GND > LOC > VIAF > Wikipedia
- Dokumentiere alle Quellen mit Authority IDs
"""


# =============================================================================
# Few-Shot Examples
# =============================================================================


FEW_SHOT_EXAMPLES_V2 = [
    {
        "input": """Die Französische Revolution begann am 14. Juli 1789 mit dem Sturm 
auf die Bastille in Paris. König Ludwig XVI. wurde am 21. Januar 1793 auf der 
Place de la Révolution hingerichtet.""",
        "reasoning": """ANALYSE:
1. Entitäten identifiziert: Französische Revolution (Event), Sturm auf die Bastille (Event), 
   Paris (Location), Ludwig XVI. (Person), Place de la Révolution (Location)
2. Daten gefunden: 14. Juli 1789, 21. Januar 1793
3. Beziehungen: Sturm war Teil der Revolution, Ludwig wurde hingerichtet

EVIDENZ-CHECK:
- "14. Juli 1789" explizit genannt ✓
- "Sturm auf die Bastille" explizit genannt ✓
- "König Ludwig XVI." explizit genannt ✓
- "21. Januar 1793" explizit genannt ✓

KONFIDENZ:
- Alle Daten explizit → 1.0
- Alle Namen vollständig → 1.0""",
        "output": {
            "nodes": [
                {
                    "node_type": "Event",
                    "name": "Französische Revolution",
                    "start_date": "1789-07-14",
                    "confidence": 1.0,
                    "text_evidence": "Die Französische Revolution begann am 14. Juli 1789",
                },
                {
                    "node_type": "Event",
                    "name": "Sturm auf die Bastille",
                    "start_date": "1789-07-14",
                    "confidence": 1.0,
                    "text_evidence": "Sturm auf die Bastille in Paris",
                },
                {
                    "node_type": "Person",
                    "name": "Ludwig XVI.",
                    "aliases": ["König Ludwig XVI."],
                    "death_date": "1793-01-21",
                    "confidence": 1.0,
                    "text_evidence": "König Ludwig XVI. wurde am 21. Januar 1793...hingerichtet",
                },
                {
                    "node_type": "Location",
                    "name": "Paris",
                    "location_type": "city",
                    "confidence": 1.0,
                    "text_evidence": "Bastille in Paris",
                },
                {
                    "node_type": "Location",
                    "name": "Place de la Révolution",
                    "parent_location": "Paris",
                    "location_type": "square",
                    "confidence": 1.0,
                    "text_evidence": "auf der Place de la Révolution hingerichtet",
                },
                {
                    "node_type": "Date",
                    "name": "14. Juli 1789",
                    "date_value": "1789-07-14",
                    "precision": "day",
                    "confidence": 1.0,
                },
                {
                    "node_type": "Date",
                    "name": "21. Januar 1793",
                    "date_value": "1793-01-21",
                    "precision": "day",
                    "confidence": 1.0,
                },
            ],
            "relationships": [
                {
                    "source_name": "Sturm auf die Bastille",
                    "target_name": "Französische Revolution",
                    "relation_type": "PART_OF",
                    "confidence": 1.0,
                },
                {
                    "source_name": "Sturm auf die Bastille",
                    "target_name": "Paris",
                    "relation_type": "LOCATED_AT",
                    "confidence": 1.0,
                },
                {
                    "source_name": "Sturm auf die Bastille",
                    "target_name": "14. Juli 1789",
                    "relation_type": "HAPPENED_ON",
                    "confidence": 1.0,
                },
                {
                    "source_name": "Ludwig XVI.",
                    "target_name": "Place de la Révolution",
                    "relation_type": "DIED_IN",
                    "confidence": 1.0,
                },
            ],
        },
    },
    {
        "input": """Der Wissenschaftler Albert Einstein entwickelte 1905 seine 
spezielle Relativitätstheorie. Er wurde 1921 mit dem Nobelpreis für Physik 
ausgezeichnet, allerdings nicht für die Relativitätstheorie, sondern für 
seine Erklärung des photoelektrischen Effekts.""",
        "reasoning": """ANALYSE:
1. Entitäten: Albert Einstein (Person), spezielle Relativitätstheorie (Konzept), 
   Nobelpreis für Physik (Event/Award), photoelektrischer Effekt (Konzept)
2. Daten: 1905, 1921
3. WICHTIG: "Konzepte" wie Theorien sind keine Events - nur die Entwicklung/Verleihung

EVIDENZ-CHECK:
- "Albert Einstein" explizit als Wissenschaftler genannt ✓
- "1905" für Relativitätstheorie explizit ✓
- "1921" für Nobelpreis explizit ✓
- Grund für Nobelpreis explizit genannt ✓

ANTI-HALLUZINATION:
- Geburtsdatum NICHT im Text → NICHT extrahieren
- Todesdatum NICHT im Text → NICHT extrahieren
- Arbeitsort NICHT im Text → NICHT extrahieren""",
        "output": {
            "nodes": [
                {
                    "node_type": "Person",
                    "name": "Albert Einstein",
                    "occupation": "Wissenschaftler",
                    "confidence": 1.0,
                    "text_evidence": "Der Wissenschaftler Albert Einstein",
                },
                {
                    "node_type": "Event",
                    "name": "Entwicklung der speziellen Relativitätstheorie",
                    "start_date": "1905-01-01",
                    "event_type": "scientific_discovery",
                    "confidence": 0.9,
                    "text_evidence": "entwickelte 1905 seine spezielle Relativitätstheorie",
                },
                {
                    "node_type": "Event",
                    "name": "Nobelpreis für Physik 1921",
                    "start_date": "1921-01-01",
                    "event_type": "award",
                    "description": "Für die Erklärung des photoelektrischen Effekts",
                    "confidence": 1.0,
                    "text_evidence": "1921 mit dem Nobelpreis für Physik ausgezeichnet",
                },
                {
                    "node_type": "Date",
                    "name": "1905",
                    "date_value": "1905-01-01",
                    "precision": "year",
                    "confidence": 1.0,
                },
                {
                    "node_type": "Date",
                    "name": "1921",
                    "date_value": "1921-01-01",
                    "precision": "year",
                    "confidence": 1.0,
                },
            ],
            "relationships": [
                {
                    "source_name": "Albert Einstein",
                    "target_name": "Entwicklung der speziellen Relativitätstheorie",
                    "relation_type": "PARTICIPATED_IN",
                    "confidence": 1.0,
                },
                {
                    "source_name": "Albert Einstein",
                    "target_name": "Nobelpreis für Physik 1921",
                    "relation_type": "PARTICIPATED_IN",
                    "confidence": 1.0,
                },
                {
                    "source_name": "Entwicklung der speziellen Relativitätstheorie",
                    "target_name": "1905",
                    "relation_type": "HAPPENED_ON",
                    "confidence": 0.9,
                },
                {
                    "source_name": "Nobelpreis für Physik 1921",
                    "target_name": "1921",
                    "relation_type": "HAPPENED_ON",
                    "confidence": 1.0,
                },
            ],
        },
    },
]


# =============================================================================
# Prompt Builder
# =============================================================================


class PromptBuilder:
    """Baut Prompts mit verschiedenen Techniken."""

    @staticmethod
    def build_extraction_prompt(
        text: str,
        use_cot: bool = True,
        use_few_shot: bool = True,
        language: str = "de",
    ) -> list[dict[str, str]]:
        """
        Baut einen vollständigen Extraction Prompt.

        Args:
            text: Zu analysierender Text
            use_cot: Chain-of-Thought aktivieren
            use_few_shot: Few-Shot Beispiele einschließen
            language: Ausgabesprache
        """
        messages = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT_V2}]

        # Few-Shot Beispiele
        if use_few_shot:
            for example in FEW_SHOT_EXAMPLES_V2[:2]:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Extrahiere den Knowledge Graph aus folgendem Text:\n\n{example['input']}",
                    }
                )

                if use_cot:
                    response = f"REASONING:\n{example['reasoning']}\n\nOUTPUT:\n{example['output']}"
                else:
                    response = str(example["output"])

                messages.append({"role": "assistant", "content": response})

        # Actual request
        user_prompt = f"""Extrahiere den Knowledge Graph aus folgendem Text.

TEXT:
{text}

ANWEISUNGEN:
1. Analysiere den Text sorgfältig (Chain-of-Thought)
2. Extrahiere NUR explizit genannte Informationen
3. Vergib realistische Konfidenzwerte
4. Gib das Ergebnis als valides JSON zurück

{"Zeige deinen Reasoning-Prozess vor dem OUTPUT." if use_cot else "Gib NUR das JSON zurück."}"""

        messages.append({"role": "user", "content": user_prompt})

        return messages

    @staticmethod
    def build_verification_prompt(
        claim_extraction: dict[str, Any],
        fact_data: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Baut einen Verification Prompt."""
        import json

        messages = [{"role": "system", "content": VERIFICATION_SYSTEM_PROMPT_V2}]

        user_prompt = f"""Verifiziere die folgende Extraktion gegen die Autoritätsdaten.

CLAIM EXTRACTION:
{json.dumps(claim_extraction, indent=2, ensure_ascii=False)}

AUTHORITY DATA (aus GND, VIAF, LOC):
{json.dumps(fact_data, indent=2, ensure_ascii=False)}

Führe das vollständige Verifikations-Protokoll durch und gib das Ergebnis als JSON zurück."""

        messages.append({"role": "user", "content": user_prompt})

        return messages

    @staticmethod
    def build_decomposition_prompt(text: str) -> list[dict[str, str]]:
        """Baut einen Fact Decomposition Prompt."""
        return [
            {"role": "system", "content": FACT_DECOMPOSITION_PROMPT},
            {"role": "user", "content": text},
        ]

    @staticmethod
    def build_consistency_prompt(
        extraction1: dict[str, Any],
        extraction2: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Baut einen Self-Consistency Check Prompt."""
        import json

        return [
            {"role": "system", "content": SELF_CONSISTENCY_PROMPT},
            {
                "role": "user",
                "content": f"""EXTRAKTION 1:
{json.dumps(extraction1, indent=2, ensure_ascii=False)}

EXTRAKTION 2:
{json.dumps(extraction2, indent=2, ensure_ascii=False)}

Analysiere die Konsistenz dieser beiden Extraktionen.""",
            },
        ]
