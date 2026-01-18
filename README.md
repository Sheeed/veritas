# The History Guardian

GraphRAG-basiertes System zur Verifikation historischer Fakten

---

## Uebersicht

The History Guardian ist ein professionelles Faktenpruefungssystem, das:

1. Historische Behauptungen analysiert mittels LLM-basierter Knowledge Graph Extraktion
2. Gegen autoritative Quellen verifiziert (GND, VIAF, LOC - keine Wikipedia)
3. ML-basierte Konfidenzwerte berechnet
4. Praezise Fehleranalysen liefert (Chronologie, Anachronismen, Widersprueche)

```
+------------------+     +-------------------+     +------------------+
|   User Input     |---->| Extraction Agent  |---->|  Claim Graph     |
|  (Text/Claim)    |     |  (GPT-4o + CoT)   |     |   (:Claim)       |
+------------------+     +-------------------+     +--------+---------+
                                                           |
+------------------+     +-------------------+             v
|  ML Confidence   |<----|  Self-Consistency |<----+------------------+
|    Scoring       |     |      Check        |     | Graph Matching   |
+------------------+     +-------------------+     |    Engine        |
                                                  +--------+---------+
                                                           |
+------------------+     +-------------------+             v
|  Verification    |<----|    Validation     |<----+------------------+
|    Result        |     |     Engine        |     | Authority Data   |
+------------------+     +-------------------+     | (GND/VIAF/LOC)   |
                                                  +------------------+
```

---

## Features

### Autoritative Datenquellen

Das System verwendet ausschliesslich bibliothekarische Normdateien - keine crowdsourced Quellen wie Wikipedia.

| Quelle    | Anbieter                     | Qualitaet | Abdeckung                         |
|-----------|------------------------------|-----------|-----------------------------------|
| GND       | Deutsche Nationalbibliothek  | Hoechste  | Deutschsprachiger Raum, universal |
| VIAF      | OCLC                         | Hoch      | International, aggregiert         |
| LOC       | Library of Congress          | Hoechste  | US-Standard, international        |
| Getty TGN | Getty Research Institute     | Hoechste  | Geografische Namen                |
| Getty ULAN| Getty Research Institute     | Hoechste  | Kuenstlernamen                    |

### AI-Techniken

- Chain-of-Thought (CoT): Strukturiertes Reasoning fuer nachvollziehbare Extraktionen
- Self-Consistency: Mehrfache Extraktion mit Konsensbildung
- Confidence Calibration: Realistische, kalibrierte Konfidenzwerte
- Anti-Hallucination Guardrails: Strikte Quellenpruefung

### Technologie-Stack

- Python 3.11+
- FastAPI (REST API)
- Streamlit (Web UI)
- Neo4j (Graph Database)
- OpenAI GPT-4o (LLM)
- Pydantic (Data Validation)

---

## Installation

### Voraussetzungen

- Python 3.11 oder hoeher
- Docker und Docker Compose
- OpenAI API Key

### Setup

```bash
# Repository klonen
git clone https://github.com/your-org/history-guardian.git
cd history-guardian

# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -e ".[dev]"

# Konfiguration
cp .env.example .env
# OPENAI_API_KEY in .env eintragen
```

### Neo4j starten

```bash
docker-compose up -d
```

### Services starten

Terminal 1 - API Server:
```bash
uvicorn src.api.main:app --reload --port 8000
```

Terminal 2 - Streamlit UI:
```bash
streamlit run src/ui/streamlit_app.py --server.port 8501
```

### Zugriff

| Service        | URL                        |
|----------------|----------------------------|
| Streamlit UI   | http://localhost:8501      |
| API Docs       | http://localhost:8000/docs |
| Neo4j Browser  | http://localhost:7474      |

Neo4j Zugangsdaten: neo4j / historyguardian2024

---

## Projektstruktur

```
history-guardian/
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI REST API
│   ├── agents/
│   │   ├── extraction.py           # Basic Extraction Agent
│   │   ├── extraction_v2.py        # Advanced Agent mit Self-Consistency
│   │   ├── prompts.py              # Basic Prompts
│   │   └── prompts_v2.py           # CoT Prompts mit Few-Shot
│   ├── db/
│   │   └── graph_db.py             # Neo4j Graph Manager
│   ├── models/
│   │   └── schema.py               # Pydantic Data Models
│   ├── datasources/
│   │   ├── authority.py            # GND, VIAF, LOC, Getty
│   │   └── external.py             # Legacy: Wikipedia, DBpedia
│   ├── validation/
│   │   └── validator.py            # Chronologie, Entity Resolution
│   ├── ml/
│   │   └── confidence.py           # Ensemble Confidence Scoring
│   ├── processing/
│   │   └── batch.py                # Batch Processing Engine
│   └── ui/
│       └── streamlit_app.py        # Streamlit Dashboard
├── tests/
│   ├── test_schema.py
│   └── test_extensions.py
├── data/                           # Ground Truth Daten
├── docker-compose.yml
├── pyproject.toml
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
└── README.md
```

---

## API Referenz

### Core Endpoints

| Endpoint              | Method | Beschreibung                |
|-----------------------|--------|-----------------------------|
| /extract              | POST   | Knowledge Graph Extraktion  |
| /ingest               | POST   | Speichern in Neo4j          |
| /validate             | POST   | Validierung gegen Facts     |
| /score/confidence     | POST   | ML Konfidenz-Score          |

### Authority Sources

| Endpoint              | Method | Beschreibung                     |
|-----------------------|--------|----------------------------------|
| /import/authority     | POST   | Import aus GND/VIAF/LOC/Getty    |
| /sources/authority    | GET    | Liste aller Autoritaetsquellen   |

### Batch Processing

| Endpoint              | Method | Beschreibung      |
|-----------------------|--------|-------------------|
| /batch/start          | POST   | Batch-Job starten |
| /batch/{id}/status    | GET    | Job-Status        |

---

## Datenmodell

### Node Types

| Typ          | Eigenschaften                              |
|--------------|--------------------------------------------|
| Person       | name, birth_date, death_date, nationality  |
| Event        | name, start_date, end_date, event_type     |
| Location     | name, location_type, coordinates           |
| Date         | date_value, precision                      |
| Organization | name, org_type, founded_date               |

### Source Labels

- :Fact - Aus Autoritaetsdatenbank (GND, VIAF, etc.)
- :Claim - Unverifizierte Behauptung

### Relationship Types

- PARTICIPATED_IN - Person nahm an Event teil
- HAPPENED_ON - Event fand an Datum statt
- LOCATED_AT - Event fand an Ort statt
- AFFILIATED_WITH - Person gehoert zu Organisation
- BORN_IN / DIED_IN - Geburts-/Sterbeort

---

## Validierungslogik

### Chronologische Pruefung

- Geburtsdatum vor Todesdatum
- Ereignisse innerhalb der Lebenszeit
- Ursache vor Wirkung
- Anachronismen-Erkennung

### Bekannte technologische Grenzdaten

| Technologie | Jahr |
|-------------|------|
| Telefon     | 1876 |
| Automobil   | 1886 |
| Flugzeug    | 1903 |
| Fernsehen   | 1927 |
| Internet    | 1983 |
| Smartphone  | 2007 |

### Entity Resolution Strategien

1. Exakt: Name identisch
2. Alias: Bekannte Namensvarianten
3. Fuzzy: Levenshtein-Aehnlichkeit
4. Phonetisch: Soundex-Matching

---

## ML Confidence Scoring

### Feature-Kategorien

Strukturelle Features:
- Anzahl Nodes/Relationships
- Vollstaendigkeit (Daten, Beschreibungen)
- Beziehungsdichte

Sprachliche Features:
- Vage Sprache ("vermutlich", "circa")
- Spezifische Zahlen/Daten

Validierungs-Features:
- Entity Match Rate
- Authority ID Coverage
- Issue Counts

### Ensemble-Modell

```
Score = 0.4 * RuleBasedScore 
      + 0.3 * LogisticRegressionScore 
      + 0.3 * RandomForestScore
```

---

## Tests

```bash
# Alle Tests ausfuehren
pytest tests/ -v

# Mit Coverage
pytest tests/ --cov=src --cov-report=html

# Nur Schema Tests
pytest tests/test_schema.py -v
```

---

## Konfiguration

### Umgebungsvariablen

| Variable         | Beschreibung              | Standard                    |
|------------------|---------------------------|-----------------------------|
| OPENAI_API_KEY   | OpenAI API Schluessel     | (erforderlich)              |
| OPENAI_MODEL     | Modell fuer Extraktion    | gpt-4o                      |
| NEO4J_URI        | Neo4j Verbindungs-URI     | bolt://localhost:7687       |
| NEO4J_USER       | Neo4j Benutzername        | neo4j                       |
| NEO4J_PASSWORD   | Neo4j Passwort            | historyguardian2024         |
| LOG_LEVEL        | Logging Level             | INFO                        |
| DEBUG            | Debug Modus               | false                       |

---

## Lizenz

MIT License - siehe LICENSE Datei

---

## Mitwirken

Beitraege sind willkommen. Siehe CONTRIBUTING.md fuer Richtlinien.

---

## Changelog

Siehe CHANGELOG.md fuer Versionshistorie.
