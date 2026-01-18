.PHONY: help install dev test lint format type-check clean docker-up docker-down api ui all

help:
	@echo "History Guardian - Verfuegbare Befehle:"
	@echo ""
	@echo "  make install      - Installiert Produktions-Dependencies"
	@echo "  make dev          - Installiert Entwicklungs-Dependencies"
	@echo "  make test         - Fuehrt alle Tests aus"
	@echo "  make lint         - Prueft Code mit ruff"
	@echo "  make format       - Formatiert Code mit black"
	@echo "  make type-check   - Prueft Typen mit mypy"
	@echo "  make clean        - Entfernt temporaere Dateien"
	@echo "  make docker-up    - Startet Neo4j Container"
	@echo "  make docker-down  - Stoppt Neo4j Container"
	@echo "  make api          - Startet FastAPI Server"
	@echo "  make ui           - Startet Streamlit UI"
	@echo "  make all          - Startet alle Services"
	@echo ""

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

docker-up:
	docker-compose up -d
	@echo "Neo4j wird gestartet..."
	@echo "Warte auf Bereitschaft..."
	@sleep 10
	@echo "Neo4j Browser: http://localhost:7474"

docker-down:
	docker-compose down

api:
	uvicorn src.api.main:app --reload --port 8000

ui:
	streamlit run src/ui/streamlit_app.py --server.port 8501

all: docker-up
	@echo "Starte Services..."
	@trap 'kill 0' EXIT; \
	uvicorn src.api.main:app --port 8000 & \
	sleep 2 && \
	streamlit run src/ui/streamlit_app.py --server.port 8501

check: lint type-check test
	@echo "Alle Pruefungen bestanden"
