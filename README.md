# 🛡️ The History Guardian

**Enterprise-grade GraphRAG-System zur Verifikation historischer Fakten**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.17+-blue.svg)](https://neo4j.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io/)

---

## 🎯 Überblick

The History Guardian ist ein professionelles Faktenprüfungssystem, das:

1. **Historische Behauptungen analysiert** mittels LLM-basierter Knowledge Graph Extraktion
2. **Gegen autoritative Quellen verifiziert** (GND, VIAF, LOC - KEINE Wikipedia!)
3. **ML-basierte Konfidenzwerte** berechnet
4. **Präzise Fehleranalysen** liefert (Chronologie, Anachronismen, Widersprüche)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Input    │────▶│ Extraction Agent │────▶│  Claim Graph    │
│  (Text/Claim)   │     │  (GPT-4o + CoT)  │     │   (:Claim)      │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              ▼
│  ML Confidence  │◀────│ Self-Consistency │◀────┌─────────────────┐
│    Scoring      │     │     Check        │     │ Graph Matching  │
└─────────────────┘     └──────────────────┘     │    Engine       │
                                                 └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              ▼
│  Verification   │◀────│   Validation     │◀────┌─────────────────┐
│    Result       │     │    Engine        │     │  Authority Data │
└─────────────────┘     └──────────────────┘     │ (GND/VIAF/LOC)  │
                                                 └─────────────────┘
```

---

## ✨ Key Features

### 🔒 Autoritative Datenquellen (KEINE crowdsourced Daten!)

| Quelle | Anbieter | Qualität | Abdeckung |
|--------|----------|----------|-----------|
| **GND** | Deutsche Nationalbibliothek | ⭐⭐⭐⭐⭐ | Deutschsprachiger Raum, universal |
| **VIAF** | OCLC | ⭐⭐⭐⭐ | International, aggregiert |
| **LOC** | Library of Congress | ⭐⭐⭐⭐⭐ | US-Standard, international |
| **Getty TGN** | Getty Research Institute | ⭐⭐⭐⭐⭐ | Geografische Namen |
| **Getty ULAN** | Getty Research Institute | ⭐⭐⭐⭐⭐ | Künstlernamen |

> ⚠️ **Warum keine Wikipedia/Wikidata?** Crowdsourced-Quellen können Fehler, Vandalismus oder veraltete Informationen enthalten. Bibliografische Normdateien werden von Fachleuten kuratiert und sind der Goldstandard in der Wissenschaft.

### 🧠 Professionelle AI-Techniken

- **Chain-of-Thought (CoT)**: Strukturiertes Reasoning für nachvollziehbare Extraktionen
- **Self-Consistency**: Mehrfache Extraktion mit Konsensbildung
- **Confidence Calibration**: Realistische, kalibrierte Konfidenzwerte
- **Anti-Hallucination Guardrails**: Strikte Quellenprüfung

### 📊 Streamlit Dashboard

Professionelle Web-Oberfläche für:
- Interaktive Faktenprüfung
- Batch-Verarbeitung
- Datenquellen-Management
- Analyse-Reports

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/your-org/history-guardian.git
cd history-guardian

# Virtuelle Umgebung
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Dependencies
pip install -e ".[dev]"

# Konfiguration
cp .env.example .env
# OPENAI_API_KEY in .env eintragen
```

### 2. Neo4j starten

```bash
docker-compose up -d
```

### 3. Services starten

```bash
# API Server
uvicorn src.api.main:app --reload --port 8000

# Streamlit UI (neues Terminal)
streamlit run src/ui/streamlit_app.py --server.port 8501
```

### 4. Zugriff

- **API Docs**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501
- **Neo4j Browser**: http://localhost:7474

---

## 📁 Projektstruktur

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
│   │   └── external.py             # (Legacy) Wikipedia, DBpedia
│   ├── validation/
│   │   └── validator.py            # Chronologie, Entity Resolution
│   ├── ml/
│   │   └── confidence.py           # Ensemble Confidence Scoring
│   ├── processing/
│   │   └── batch.py                # Batch Processing Engine
│   └── ui/
│       └── streamlit_app.py        # Streamlit Dashboard
├── tests/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 🔧 API Endpoints

### Core Endpoints

| Endpoint | Method | Beschreibung |
|----------|--------|--------------|
| `POST /extract` | POST | Knowledge Graph Extraktion |
| `POST /ingest` | POST | Speichern in Neo4j |
| `POST /validate` | POST | Validierung gegen Facts |
| `POST /score/confidence` | POST | ML Konfidenz-Score |

### Authority Sources

| Endpoint | Method | Beschreibung |
|----------|--------|--------------|
| `POST /import/authority` | POST | Import aus GND/VIAF/LOC/Getty |
| `GET /sources/authority` | GET | Liste aller Autoritätsquellen |

### Batch Processing

| Endpoint | Method | Beschreibung |
|----------|--------|--------------|
| `POST /batch/start` | POST | Batch-Job starten |
| `GET /batch/{id}/status` | GET | Job-Status |

---

## 🔬 Validierungslogik

### Chronologische Prüfung

```
✓ Geburtsdatum vor Todesdatum
✓ Ereignisse innerhalb der Lebenszeit
✓ Ursache vor Wirkung
✗ Anachronismen (z.B. "telefonierte 1850")
```

### Anachronismus-Erkennung

Bekannte technologische Grenzdaten:
- Telefon: 1876
- Automobil: 1886
- Flugzeug: 1903
- Fernsehen: 1927
- Internet: 1983
- Smartphone: 2007

### Entity Resolution

1. **Exakt**: Name identisch
2. **Alias**: Bekannte Namensvarianten
3. **Fuzzy**: Levenshtein-Ähnlichkeit
4. **Phonetisch**: Soundex-Matching

---

## 🤖 ML Confidence Scoring

### Feature-Kategorien

**Strukturelle Features:**
- Anzahl Nodes/Relationships
- Vollständigkeit (Daten, Beschreibungen)
- Beziehungsdichte

**Sprachliche Features:**
- Vage Sprache ("vermutlich", "circa")
- Spezifische Zahlen/Daten
- Namensqualität

**Validierungs-Features:**
- Entity Match Rate
- Authority ID Coverage
- Issue Counts

### Ensemble-Modell

```
Score = 0.4 × RuleBasedScore 
      + 0.3 × LogisticRegressionScore 
      + 0.3 × RandomForestScore
```

---

## 📊 Datenmodell

### Node Types

| Typ | Eigenschaften | Beispiel |
|-----|--------------|----------|
| Person | name, birth_date, death_date, nationality | Napoleon Bonaparte |
| Event | name, start_date, end_date, event_type | Französische Revolution |
| Location | name, location_type, coordinates | Paris |
| Date | date_value, precision | 1789-07-14 |
| Organization | name, org_type, founded_date | Académie française |

### Source Labels

- `:Fact` - Aus Autoritätsdatenbank (GND, VIAF, etc.)
- `:Claim` - Unverifizierte Behauptung

---

## 🧪 Tests

```bash
# Unit Tests
pytest tests/ -v

# Mit Coverage
pytest tests/ --cov=src --cov-report=html

# Nur Extraction Tests
pytest tests/test_extraction.py -v
```

---

## 🔒 Best Practices

### Für höchste Datenqualität:

1. **Immer Authority Sources verwenden** - keine Wikipedia/Wikidata
2. **Self-Consistency aktivieren** für kritische Anwendungen
3. **Konfidenz-Schwellenwerte** setzen (empfohlen: ≥0.8 für Fakten)
4. **Regelmäßige Evaluation** gegen Ground Truth

### Für Performance:

1. **Batch-Processing** für große Mengen
2. **Caching** für häufige Authority-Anfragen
3. **Connection Pooling** für Neo4j

---

## 📈 Roadmap

- [ ] Weitere Autoritätsquellen (BnF, British Library)
- [ ] Temporal Reasoning Engine
- [ ] Graph Visualization
- [ ] Fine-tuned Extraction Model
- [ ] Multi-Language Support
- [ ] Redis Job Queue

---

## 📄 Lizenz

MIT License

---

## 🤝 Contributing

Contributions sind willkommen! Bitte beachte:
- Code muss mit `ruff` und `mypy` validiert werden
- Tests für neue Features erforderlich
- Dokumentation aktualisieren

---

**Built with ❤️ for historical accuracy**

*"In der Geschichte gibt es keine Meinungen, nur Fakten."*
