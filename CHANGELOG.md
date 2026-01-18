# Changelog

Alle wesentlichen Aenderungen an diesem Projekt werden in dieser Datei dokumentiert.

Das Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/1.0.0/),
und dieses Projekt folgt [Semantic Versioning](https://semver.org/lang/de/).

---

## [Unreleased]

### Geplant
- Weitere Autoritaetsquellen (BnF, British Library)
- Graph Visualization im UI
- Multi-Language Support
- Fine-tuned Extraction Model

---

## [0.3.0] - 2024-XX-XX

### Hinzugefuegt
- Streamlit Web UI als Ersatz fuer React
- Autoritative Datenquellen (GND, VIAF, LOC, Getty TGN, Getty ULAN)
- Chain-of-Thought (CoT) Prompting
- Self-Consistency Checking fuer robustere Extraktionen
- Extraction Agent V2 mit Anti-Hallucination Guardrails
- Professionelle Few-Shot Beispiele
- API Endpoint: POST /import/authority
- API Endpoint: GET /sources/authority

### Geaendert
- Datenquellen von Wikipedia/Wikidata auf bibliothekarische Normdateien umgestellt
- Prompts ueberarbeitet fuer bessere Konfidenz-Kalibrierung
- README komplett ueberarbeitet

### Entfernt
- React-basierte Web UI (ersetzt durch Streamlit)
- Wikipedia als primaere Datenquelle (nur noch als Legacy verfuegbar)

---

## [0.2.0] - 2024-XX-XX

### Hinzugefuegt
- Batch-Verarbeitung mit paralleler Ausfuehrung
- Fortgeschrittene Validierungslogik
  - Entity Resolution (Exakt, Alias, Fuzzy, Phonetisch)
  - Chronologische Konsistenzpruefung
  - Anachronismus-Erkennung
- ML Confidence Scoring mit Ensemble-Modell
  - Regelbasierter Scorer
  - Logistische Regression
  - Random Forest
- Feature Extraction (25 Features)
- Externe Datenquellen (Wikidata, DBpedia)
- React Web UI
- API Endpoints fuer Batch und Validation
- Progress Tracking fuer Batch Jobs

### Geaendert
- API Response Models erweitert
- Graph Manager um Fact-Import erweitert

---

## [0.1.0] - 2024-XX-XX

### Hinzugefuegt
- Initiales Projekt-Scaffold
- Pydantic Data Models (Person, Event, Location, Date, Organization)
- Neo4j Graph Database Integration
- Docker Compose Setup
- LLM Extraction Agent mit OpenAI GPT-4o
- FastAPI REST API
  - POST /extract - Knowledge Graph Extraktion
  - POST /ingest - Graph Speicherung
  - GET /stats - Statistiken
  - DELETE /claims - Claims loeschen
  - GET /health - Health Check
- Claim vs. Fact Unterscheidung (:Claim, :Fact Labels)
- Basis-Tests fuer Schema Validation
- Projektdokumentation

---

## Versionsschema

- MAJOR: Inkompatible API-Aenderungen
- MINOR: Neue Funktionalitaet (abwaertskompatibel)
- PATCH: Fehlerbehebungen (abwaertskompatibel)
