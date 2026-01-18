# Mitwirken bei The History Guardian

Vielen Dank fuer dein Interesse an diesem Projekt. Diese Anleitung erklaert, wie du beitragen kannst.

---

## Verhaltenskodex

Dieses Projekt folgt einem offenen und respektvollen Umgang. Wir erwarten von allen Beteiligten:

- Respektvolle und konstruktive Kommunikation
- Akzeptanz unterschiedlicher Meinungen und Erfahrungen
- Fokus auf das Beste fuer die Gemeinschaft
- Empathie gegenueber anderen Mitwirkenden

---

## Wie kann ich beitragen?

### Fehler melden

1. Pruefe zuerst, ob der Fehler bereits gemeldet wurde (Issues durchsuchen)
2. Erstelle ein neues Issue mit:
   - Beschreibung des Problems
   - Schritte zur Reproduktion
   - Erwartetes vs. tatsaechliches Verhalten
   - System-Informationen (OS, Python-Version, etc.)

### Feature-Vorschlaege

1. Erstelle ein Issue mit dem Label "enhancement"
2. Beschreibe das gewuenschte Feature detailliert
3. Erklaere den Anwendungsfall

### Code beitragen

1. Forke das Repository
2. Erstelle einen Feature-Branch: `git checkout -b feature/mein-feature`
3. Implementiere deine Aenderungen
4. Schreibe Tests fuer neue Funktionalitaet
5. Stelle sicher, dass alle Tests bestehen
6. Committe mit aussagekraeftiger Nachricht
7. Push zum Fork: `git push origin feature/mein-feature`
8. Erstelle einen Pull Request

---

## Entwicklungsrichtlinien

### Code-Stil

- Python: PEP 8 konform
- Formatierung mit `black`
- Linting mit `ruff`
- Type Hints verwenden (mypy kompatibel)

### Vor dem Commit

```bash
# Formatierung
black src/ tests/

# Linting
ruff check src/ tests/

# Type Checking
mypy src/

# Tests
pytest tests/ -v
```

### Commit-Nachrichten

Format: `<typ>: <beschreibung>`

Typen:
- feat: Neues Feature
- fix: Fehlerbehebung
- docs: Dokumentationsaenderung
- style: Formatierung (kein Code-Aenderung)
- refactor: Code-Umstrukturierung
- test: Tests hinzugefuegt/geaendert
- chore: Build-Prozess, Dependencies

Beispiele:
```
feat: Add Getty TGN data source integration
fix: Correct date parsing for BC dates
docs: Update API documentation
test: Add validation tests for chronology checker
```

### Branch-Namenskonvention

- feature/beschreibung - Neue Features
- fix/beschreibung - Fehlerbehebungen
- docs/beschreibung - Dokumentation
- refactor/beschreibung - Umstrukturierungen

---

## Pull Request Prozess

1. Aktualisiere die README.md falls noetig
2. Aktualisiere CHANGELOG.md mit deinen Aenderungen
3. Stelle sicher, dass alle CI-Checks bestehen
4. Ein Maintainer wird den PR reviewen
5. Nach Genehmigung wird der PR gemergt

### PR Checkliste

- [ ] Code folgt den Stilrichtlinien
- [ ] Tests wurden hinzugefuegt/aktualisiert
- [ ] Dokumentation wurde aktualisiert
- [ ] CHANGELOG.md wurde aktualisiert
- [ ] Alle CI-Checks bestehen

---

## Projektstruktur fuer Beitraege

### Neue Datenquelle hinzufuegen

1. Erstelle neue Klasse in `src/datasources/`
2. Erbe von `AuthoritySource`
3. Implementiere `search()` und `get_by_id()`
4. Registriere in `AuthoritySourceManager`
5. Fuege Tests hinzu
6. Dokumentiere in README.md

### Neue Validierungsregel hinzufuegen

1. Erweitere `src/validation/validator.py`
2. Definiere neuen `IssueType` falls noetig
3. Implementiere Prueflogik
4. Fuege Tests hinzu

### Neue API Endpoints hinzufuegen

1. Definiere Request/Response Models in `main.py`
2. Implementiere Endpoint-Funktion
3. Fuege OpenAPI-Dokumentation hinzu
4. Fuege Tests hinzu

---

## Tests schreiben

### Unit Tests

```python
def test_entity_resolver_exact_match():
    """Test exaktes Entity Matching."""
    resolver = EntityResolver()
    # ... test implementation
    assert result.match_method == "exact"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_gnd_search():
    """Test GND API Integration."""
    source = GNDSource()
    results = await source.search("Napoleon", NodeType.PERSON)
    assert len(results) > 0
    await source.close()
```

---

## Fragen?

Bei Fragen erstelle ein Issue mit dem Label "question" oder kontaktiere die Maintainer.
