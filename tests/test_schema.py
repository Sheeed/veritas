"""Tests für das Schema-Modul."""

from datetime import date

import pytest

from src.models.schema import (
    EventNode,
    KnowledgeGraphExtraction,
    NodeType,
    PersonNode,
    Relationship,
    RelationType,
    SourceLabel,
)


class TestPersonNode:
    """Tests für PersonNode."""

    def test_create_minimal_person(self):
        """Test: Minimale Person-Erstellung."""
        person = PersonNode(name="Napoleon Bonaparte")

        assert person.name == "Napoleon Bonaparte"
        assert person.node_type == NodeType.PERSON
        assert person.source_label == SourceLabel.CLAIM
        assert person.confidence == 1.0

    def test_create_full_person(self):
        """Test: Vollständige Person-Erstellung."""
        person = PersonNode(
            name="Napoleon Bonaparte",
            aliases=["Napoleon", "Bonaparte"],
            birth_date=date(1769, 8, 15),
            death_date=date(1821, 5, 5),
            nationality="French",
            occupation=["Emperor", "Military Leader"],
        )

        assert person.birth_date == date(1769, 8, 15)
        assert "Napoleon" in person.aliases
        assert "Emperor" in person.occupation


class TestEventNode:
    """Tests für EventNode."""

    def test_date_validation(self):
        """Test: End-Datum muss nach Start-Datum liegen."""
        # Valider Fall
        event = EventNode(
            name="Battle of Waterloo",
            start_date=date(1815, 6, 18),
            end_date=date(1815, 6, 18),
        )
        assert event.start_date == event.end_date

    def test_invalid_date_range(self):
        """Test: Ungültiger Datumsbereich wirft Fehler."""
        with pytest.raises(ValueError, match="end_date must be after start_date"):
            EventNode(
                name="Invalid Event",
                start_date=date(1815, 6, 18),
                end_date=date(1815, 6, 17),  # Vor start_date!
            )


class TestRelationship:
    """Tests für Relationship."""

    def test_create_relationship(self):
        """Test: Beziehung erstellen."""
        rel = Relationship(
            source_name="Napoleon Bonaparte",
            source_type=NodeType.PERSON,
            relation_type=RelationType.PARTICIPATED_IN,
            target_name="Battle of Waterloo",
            target_type=NodeType.EVENT,
        )

        assert rel.relation_type == RelationType.PARTICIPATED_IN
        assert rel.confidence == 1.0

    def test_name_normalization(self):
        """Test: Namen werden normalisiert (getrimmt)."""
        rel = Relationship(
            source_name="  Napoleon  ",
            source_type=NodeType.PERSON,
            relation_type=RelationType.PARTICIPATED_IN,
            target_name="  Waterloo  ",
            target_type=NodeType.EVENT,
        )

        assert rel.source_name == "Napoleon"
        assert rel.target_name == "Waterloo"


class TestKnowledgeGraphExtraction:
    """Tests für KnowledgeGraphExtraction."""

    def test_get_triples(self):
        """Test: Triple-Extraktion."""
        extraction = KnowledgeGraphExtraction(
            nodes=[
                PersonNode(name="Napoleon"),
                EventNode(name="Waterloo"),
            ],
            relationships=[
                Relationship(
                    source_name="Napoleon",
                    source_type=NodeType.PERSON,
                    relation_type=RelationType.PARTICIPATED_IN,
                    target_name="Waterloo",
                    target_type=NodeType.EVENT,
                )
            ],
        )

        triples = extraction.get_triples()
        assert len(triples) == 1
        assert triples[0] == ("Napoleon", "PARTICIPATED_IN", "Waterloo")

    def test_get_nodes_by_type(self):
        """Test: Nodes nach Typ filtern."""
        extraction = KnowledgeGraphExtraction(
            nodes=[
                PersonNode(name="Napoleon"),
                PersonNode(name="Wellington"),
                EventNode(name="Waterloo"),
            ],
        )

        persons = extraction.get_nodes_by_type(NodeType.PERSON)
        assert len(persons) == 2

        events = extraction.get_nodes_by_type(NodeType.EVENT)
        assert len(events) == 1
