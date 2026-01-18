"""
Veritas Propaganda Pattern Detector

Erkennt Propaganda-Muster und manipulative Narrative in historischen Behauptungen.

Features:
- Pattern Matching gegen bekannte Propaganda-Techniken
- Sprachanalyse für manipulative Rhetorik
- Narrative Pattern Detection
- Emotionale Sprache Erkennung
"""

import logging
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Propaganda Techniques
# =============================================================================

class PropagandaTechnique(str, Enum):
    """Bekannte Propaganda-Techniken."""
    
    # Logische Fehler
    FALSE_DILEMMA = "false_dilemma"  # Nur 2 Optionen präsentieren
    SLIPPERY_SLOPE = "slippery_slope"  # Übertriebene Konsequenzen
    STRAW_MAN = "straw_man"  # Argument verzerren
    AD_HOMINEM = "ad_hominem"  # Person angreifen statt Argument
    
    # Emotionale Manipulation
    APPEAL_TO_FEAR = "appeal_to_fear"  # Angst schüren
    APPEAL_TO_AUTHORITY = "appeal_to_authority"  # Falsche Autoritäten
    APPEAL_TO_EMOTION = "appeal_to_emotion"  # Emotionen statt Fakten
    BANDWAGON = "bandwagon"  # Alle machen es
    
    # Verzerrung
    CHERRY_PICKING = "cherry_picking"  # Selektive Fakten
    WHATABOUTISM = "whataboutism"  # Ablenken auf andere
    FALSE_EQUIVALENCE = "false_equivalence"  # Ungleiches gleichsetzen
    LOADED_LANGUAGE = "loaded_language"  # Emotionale Wörter
    
    # Historische Manipulation
    REVISIONISM = "revisionism"  # Geschichte umschreiben
    VICTIM_NARRATIVE = "victim_narrative"  # Opferrolle beanspruchen
    HERO_WORSHIP = "hero_worship"  # Überhöhung von Personen
    ENEMY_IMAGE = "enemy_image"  # Feindbilder aufbauen
    
    # Desinformation
    FABRICATION = "fabrication"  # Erfundene Fakten
    OMISSION = "omission"  # Wichtiges weglassen
    EXAGGERATION = "exaggeration"  # Übertreibung


class PropagandaIndicator(BaseModel):
    """Ein erkannter Propaganda-Indikator."""
    
    technique: PropagandaTechnique
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str  # Was wurde gefunden
    explanation: str  # Warum ist das problematisch
    text_span: Optional[str] = None  # Betroffener Textabschnitt


class PropagandaAnalysis(BaseModel):
    """Ergebnis der Propaganda-Analyse."""
    
    is_propaganda: bool = False
    propaganda_score: float = Field(ge=0.0, le=1.0, default=0.0)
    risk_level: str = "low"  # low, medium, high, critical
    
    detected_techniques: List[PropagandaIndicator] = Field(default_factory=list)
    narrative_patterns: List[str] = Field(default_factory=list)
    
    summary: str = ""
    recommendation: str = ""


# =============================================================================
# Pattern Definitions
# =============================================================================

@dataclass
class LanguagePattern:
    """Ein sprachliches Muster für Propaganda-Erkennung."""
    
    technique: PropagandaTechnique
    patterns: List[str]  # Regex patterns
    explanation: str
    weight: float = 1.0


# Definierte Muster für verschiedene Techniken
PROPAGANDA_PATTERNS: List[LanguagePattern] = [
    # Appeal to Fear
    LanguagePattern(
        technique=PropagandaTechnique.APPEAL_TO_FEAR,
        patterns=[
            r'\b(threat|danger|catastroph|disast|destroy|annihilat|exterminat)\w*\b',
            r'\b(bedroh|gefahr|katastroph|vernicht|zerstör|auslösch)\w*\b',
            r'\b(if we don\'t|wenn wir nicht|otherwise|sonst)\b.*\b(will|werden)\b',
        ],
        explanation="Uses fear-inducing language to manipulate",
        weight=1.2,
    ),
    
    # Loaded Language
    LanguagePattern(
        technique=PropagandaTechnique.LOADED_LANGUAGE,
        patterns=[
            r'\b(evil|wicked|sinister|vile|despicable)\b',
            r'\b(böse|teuflisch|niederträchtig|verabscheuungswürdig)\b',
            r'\b(heroic|glorious|magnificent|noble|righteous)\b',
            r'\b(heldenhaft|glorreich|edel|rechtschaffen)\b',
            r'\b(traitor|verräter|enemy of the people|volksfeind)\b',
        ],
        explanation="Uses emotionally charged words instead of neutral terms",
        weight=1.0,
    ),
    
    # False Dilemma
    LanguagePattern(
        technique=PropagandaTechnique.FALSE_DILEMMA,
        patterns=[
            r'\b(either|or|entweder|oder)\b.*\b(only|nur|no other|kein anderer)\b',
            r'\b(you\'re (either )?with us or against)\b',
            r'\b(there (is|are) only two|es gibt nur zwei)\b',
        ],
        explanation="Presents only two options when more exist",
        weight=1.1,
    ),
    
    # Appeal to Authority
    LanguagePattern(
        technique=PropagandaTechnique.APPEAL_TO_AUTHORITY,
        patterns=[
            r'\b(experts say|wissenschaftler sagen|everyone knows|jeder weiß)\b',
            r'\b(it is well known|es ist bekannt|obviously|offensichtlich)\b',
            r'\b(studies show|studien zeigen)\b(?!.*\bsource\b)',
        ],
        explanation="Appeals to unnamed authorities without evidence",
        weight=0.9,
    ),
    
    # Bandwagon
    LanguagePattern(
        technique=PropagandaTechnique.BANDWAGON,
        patterns=[
            r'\b(everyone|everybody|all people|alle menschen|jeder)\b.*\b(know|believe|think|wissen|glauben|denken)\b',
            r'\b(millions of|millionen von|the majority|die mehrheit)\b',
            r'\b(no one|niemand|nobody)\b.*\b(disputes|bestreitet|doubts|bezweifelt)\b',
        ],
        explanation="Implies everyone agrees to pressure conformity",
        weight=0.8,
    ),
    
    # Whataboutism
    LanguagePattern(
        technique=PropagandaTechnique.WHATABOUTISM,
        patterns=[
            r'\b(what about|was ist mit|but they also|aber die haben auch)\b',
            r'\b(but.*did the same|aber.*das gleiche getan)\b',
            r'\b(you should look at|man sollte sich.*ansehen)\b.*\b(instead|stattdessen)\b',
        ],
        explanation="Deflects criticism by pointing to others' faults",
        weight=1.0,
    ),
    
    # Revisionism
    LanguagePattern(
        technique=PropagandaTechnique.REVISIONISM,
        patterns=[
            r'\b(the (real|true) history|die (wahre|echte) geschichte)\b',
            r'\b(mainstream historians|mainstream-historiker)\b.*\b(lie|wrong|lügen|falsch)\b',
            r'\b(covered up|vertuscht|suppressed|unterdrückt)\b.*\b(truth|wahrheit)\b',
            r'\b(they don\'t want you to know|das will man uns nicht sagen)\b',
        ],
        explanation="Claims to reveal 'hidden' history against established facts",
        weight=1.3,
    ),
    
    # Victim Narrative
    LanguagePattern(
        technique=PropagandaTechnique.VICTIM_NARRATIVE,
        patterns=[
            r'\b(persecuted|verfolgt|oppressed|unterdrückt|victimized|zum opfer gemacht)\b',
            r'\b(we are|wir sind)\b.*\b(victims|opfer)\b',
            r'\b(attacked|angegriffen|targeted|ins visier genommen)\b.*\b(unfairly|zu unrecht)\b',
        ],
        explanation="Claims victimhood to gain sympathy or deflect criticism",
        weight=1.1,
    ),
    
    # Enemy Image
    LanguagePattern(
        technique=PropagandaTechnique.ENEMY_IMAGE,
        patterns=[
            r'\b(the (jews|muslims|christians|foreigners)|die (juden|muslime|christen|ausländer))\b.*\b(want|wollen|control|kontrollieren)\b',
            r'\b(conspiracy|verschwörung)\b.*\b(against|gegen)\b',
            r'\b(they|sie)\b.*\b(want to destroy|wollen zerstören|are trying to|versuchen zu)\b',
        ],
        explanation="Creates or reinforces enemy stereotypes",
        weight=1.4,
    ),
    
    # Exaggeration
    LanguagePattern(
        technique=PropagandaTechnique.EXAGGERATION,
        patterns=[
            r'\b(always|never|every single|immer|nie|jeder einzelne)\b',
            r'\b(the (biggest|worst|greatest|best) in history|der (größte|schlimmste|beste) in der geschichte)\b',
            r'\b(unprecedented|noch nie dagewesen|unparalleled|beispiellos)\b',
            r'\b(100%|completely|totally|völlig|komplett|absolut)\b.*\b(wrong|false|falsch)\b',
        ],
        explanation="Uses extreme language that overstates the case",
        weight=0.9,
    ),
    
    # Hero Worship
    LanguagePattern(
        technique=PropagandaTechnique.HERO_WORSHIP,
        patterns=[
            r'\b(genius|brilliant|visionary|genie|brillant|visionär)\b.*\b(leader|führer|figure|figur)\b',
            r'\b(saved|gerettet|single-handedly|im alleingang)\b',
            r'\b(great (man|leader)|großer (mann|führer))\b',
            r'\b(without (him|her)|ohne (ihn|sie))\b.*\b(would have|hätte)\b',
        ],
        explanation="Attributes excessive credit to a single person",
        weight=1.0,
    ),
]


# Known propaganda narratives
KNOWN_NARRATIVES = {
    "clean_army": {
        "keywords": ["clean", "sauber", "regular army", "reguläre armee", "only following orders", "nur befehle befolgt", "not involved", "nicht beteiligt"],
        "description": "Claims regular military was not involved in atrocities",
    },
    "victim_numbers": {
        "keywords": ["real number", "wahre zahl", "actually died", "tatsächlich gestorben", "covered up", "vertuscht", "exaggerated", "übertrieben"],
        "description": "Disputes established death tolls without evidence",
    },
    "stab_in_back": {
        "keywords": ["stab in the back", "dolchstoß", "betrayed", "verraten", "from within", "von innen"],
        "description": "Claims defeat was caused by internal betrayal",
    },
    "golden_age": {
        "keywords": ["used to be", "früher war", "in the old days", "in alten zeiten", "before", "bevor", "destroyed", "zerstört"],
        "description": "Idealizes a past era and blames its end on others",
    },
    "hidden_truth": {
        "keywords": ["they don't want", "die wollen nicht", "hidden", "verborgen", "secret", "geheim", "the truth is", "die wahrheit ist"],
        "description": "Claims important truths are being suppressed",
    },
}


# =============================================================================
# Propaganda Detector
# =============================================================================

class PropagandaDetector:
    """
    Erkennt Propaganda-Muster in Texten.
    
    Kombiniert:
    - Regelbasiertes Pattern Matching
    - Bekannte Narrative-Erkennung
    - Sprachanalyse
    """
    
    def __init__(self):
        self.patterns = PROPAGANDA_PATTERNS
        self.narratives = KNOWN_NARRATIVES
    
    def analyze(self, text: str) -> PropagandaAnalysis:
        """
        Analysiert einen Text auf Propaganda-Muster.
        
        Args:
            text: Zu analysierender Text
            
        Returns:
            PropagandaAnalysis mit Details
        """
        text_lower = text.lower()
        
        detected_techniques: List[PropagandaIndicator] = []
        narrative_patterns: List[str] = []
        
        # === Pattern Matching ===
        for lang_pattern in self.patterns:
            for pattern in lang_pattern.patterns:
                try:
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    if matches:
                        # Confidence basierend auf Anzahl der Matches
                        confidence = min(1.0, 0.5 + 0.1 * len(matches)) * lang_pattern.weight
                        
                        indicator = PropagandaIndicator(
                            technique=lang_pattern.technique,
                            confidence=round(confidence, 2),
                            evidence=f"Found pattern: {matches[0] if isinstance(matches[0], str) else matches[0][0]}",
                            explanation=lang_pattern.explanation,
                            text_span=matches[0] if isinstance(matches[0], str) else str(matches[0]),
                        )
                        
                        # Nur hinzufügen wenn nicht schon gleiche Technik mit höherer Confidence
                        existing = [d for d in detected_techniques if d.technique == lang_pattern.technique]
                        if not existing or existing[0].confidence < confidence:
                            if existing:
                                detected_techniques.remove(existing[0])
                            detected_techniques.append(indicator)
                            
                except re.error as e:
                    logger.warning(f"Invalid regex pattern: {e}")
        
        # === Narrative Detection ===
        for narrative_id, narrative in self.narratives.items():
            keyword_matches = sum(1 for kw in narrative["keywords"] if kw in text_lower)
            if keyword_matches >= 2:  # Mindestens 2 Keywords
                narrative_patterns.append(f"{narrative_id}: {narrative['description']}")
        
        # === Score berechnen ===
        if detected_techniques:
            raw_score = sum(d.confidence for d in detected_techniques) / len(detected_techniques)
            # Bonus für mehrere Techniken
            technique_bonus = min(0.3, 0.1 * len(detected_techniques))
            # Bonus für bekannte Narrative
            narrative_bonus = min(0.2, 0.1 * len(narrative_patterns))
            
            propaganda_score = min(1.0, raw_score + technique_bonus + narrative_bonus)
        else:
            propaganda_score = 0.0
        
        # === Risk Level ===
        if propaganda_score >= 0.8:
            risk_level = "critical"
        elif propaganda_score >= 0.6:
            risk_level = "high"
        elif propaganda_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # === Summary & Recommendation ===
        is_propaganda = propaganda_score >= 0.5
        
        if is_propaganda:
            techniques_str = ", ".join(set(d.technique.value for d in detected_techniques[:3]))
            summary = f"This text shows signs of propaganda ({risk_level} risk). Detected techniques: {techniques_str}."
            
            if narrative_patterns:
                summary += f" Matches known narrative patterns: {narrative_patterns[0]}."
            
            recommendation = "This content should be fact-checked against authoritative sources. Be cautious of emotional manipulation and verify all claims independently."
        else:
            summary = "No significant propaganda patterns detected."
            recommendation = "Standard fact-checking procedures apply."
        
        return PropagandaAnalysis(
            is_propaganda=is_propaganda,
            propaganda_score=round(propaganda_score, 3),
            risk_level=risk_level,
            detected_techniques=detected_techniques,
            narrative_patterns=narrative_patterns,
            summary=summary,
            recommendation=recommendation,
        )
    
    def quick_check(self, text: str) -> Dict[str, Any]:
        """
        Schnelle Propaganda-Prüfung für UI.
        
        Returns:
            Vereinfachtes Ergebnis-Dict
        """
        analysis = self.analyze(text)
        
        return {
            "is_propaganda": analysis.is_propaganda,
            "score": analysis.propaganda_score,
            "risk_level": analysis.risk_level,
            "techniques_found": len(analysis.detected_techniques),
            "summary": analysis.summary,
        }
    
    def get_technique_info(self, technique: PropagandaTechnique) -> Dict[str, str]:
        """Gibt Info über eine Propaganda-Technik zurück."""
        descriptions = {
            PropagandaTechnique.FALSE_DILEMMA: "Presenting only two options when more exist, forcing a false choice.",
            PropagandaTechnique.SLIPPERY_SLOPE: "Claiming one event will lead to extreme consequences without evidence.",
            PropagandaTechnique.STRAW_MAN: "Misrepresenting an argument to make it easier to attack.",
            PropagandaTechnique.AD_HOMINEM: "Attacking the person making the argument instead of the argument itself.",
            PropagandaTechnique.APPEAL_TO_FEAR: "Using fear to manipulate emotions and bypass rational thinking.",
            PropagandaTechnique.APPEAL_TO_AUTHORITY: "Citing authority figures without proper evidence or credentials.",
            PropagandaTechnique.APPEAL_TO_EMOTION: "Using emotional appeals instead of logical arguments.",
            PropagandaTechnique.BANDWAGON: "Implying something is true because many people believe it.",
            PropagandaTechnique.CHERRY_PICKING: "Selecting only evidence that supports a position while ignoring contrary evidence.",
            PropagandaTechnique.WHATABOUTISM: "Deflecting criticism by pointing to others' wrongdoings.",
            PropagandaTechnique.FALSE_EQUIVALENCE: "Treating two things as equivalent when they are not.",
            PropagandaTechnique.LOADED_LANGUAGE: "Using emotionally charged words to influence opinion.",
            PropagandaTechnique.REVISIONISM: "Rewriting history to support a particular narrative.",
            PropagandaTechnique.VICTIM_NARRATIVE: "Claiming victimhood to gain sympathy or deflect criticism.",
            PropagandaTechnique.HERO_WORSHIP: "Excessive glorification of a person or group.",
            PropagandaTechnique.ENEMY_IMAGE: "Creating or reinforcing negative stereotypes about a group.",
            PropagandaTechnique.FABRICATION: "Making up facts or events that never happened.",
            PropagandaTechnique.OMISSION: "Deliberately leaving out important information.",
            PropagandaTechnique.EXAGGERATION: "Overstating facts or claims beyond what evidence supports.",
        }
        
        return {
            "name": technique.value.replace("_", " ").title(),
            "description": descriptions.get(technique, "No description available."),
        }


# =============================================================================
# Singleton
# =============================================================================

_detector_instance: Optional[PropagandaDetector] = None


def get_propaganda_detector() -> PropagandaDetector:
    """Gibt Propaganda Detector Instanz zurück."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PropagandaDetector()
    return _detector_instance