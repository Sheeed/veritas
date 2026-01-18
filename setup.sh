#!/bin/bash
# Setup-Skript fuer History Guardian
# Verwendung: ./setup.sh

set -e

echo "=========================================="
echo "History Guardian - Setup"
echo "=========================================="
echo ""

# Pruefe Python Version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "FEHLER: Python $REQUIRED_VERSION oder hoeher erforderlich (gefunden: $PYTHON_VERSION)"
    exit 1
fi
echo "[OK] Python Version: $PYTHON_VERSION"

# Pruefe Docker
if ! command -v docker &> /dev/null; then
    echo "WARNUNG: Docker nicht gefunden. Neo4j muss manuell installiert werden."
else
    echo "[OK] Docker gefunden"
fi

# Erstelle virtuelle Umgebung
if [ ! -d "venv" ]; then
    echo ""
    echo "Erstelle virtuelle Umgebung..."
    python3 -m venv venv
fi

# Aktiviere virtuelle Umgebung
source venv/bin/activate

# Installiere Dependencies
echo ""
echo "Installiere Dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# Erstelle .env falls nicht vorhanden
if [ ! -f ".env" ]; then
    echo ""
    echo "Erstelle .env Datei..."
    cp .env.example .env
    echo "WICHTIG: Bitte OPENAI_API_KEY in .env eintragen!"
fi

# Starte Neo4j falls Docker verfuegbar
if command -v docker &> /dev/null; then
    echo ""
    echo "Starte Neo4j..."
    docker-compose up -d
    echo "Warte auf Neo4j..."
    sleep 10
fi

echo ""
echo "=========================================="
echo "Setup abgeschlossen!"
echo "=========================================="
echo ""
echo "Naechste Schritte:"
echo "1. OPENAI_API_KEY in .env eintragen"
echo "2. source venv/bin/activate"
echo "3. make api    (Terminal 1)"
echo "4. make ui     (Terminal 2)"
echo ""
echo "URLs:"
echo "- API:     http://localhost:8000/docs"
echo "- UI:      http://localhost:8501"
echo "- Neo4j:   http://localhost:7474"
echo ""
