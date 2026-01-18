"""
Neo4j Graph Database Manager.

Handhabt alle Interaktionen mit der Neo4j Datenbank für
Knowledge Graph Speicherung und Abfragen.
"""

import logging
from contextlib import asynccontextmanager
from datetime import date
from typing import Any
from uuid import UUID

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.config import get_settings
from src.models.schema import (
    AnyNode,
    DateNode,
    EventNode,
    KnowledgeGraphExtraction,
    LocationNode,
    NodeType,
    OrganizationNode,
    PersonNode,
    Relationship,
    SourceLabel,
)

logger = logging.getLogger(__name__)


class GraphManager:
    """
    Manager für Neo4j Graph-Operationen.

    Unterstützt:
    - Verbindungsmanagement mit Connection Pooling
    - CRUD-Operationen für Nodes und Relationships
    - Trennung von :Claim und :Fact Labels
    - Batch-Operationen für Performance
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        """
        Initialisiert den GraphManager.

        Args:
            uri: Neo4j Bolt URI (default: aus Settings)
            user: Neo4j Benutzername (default: aus Settings)
            password: Neo4j Passwort (default: aus Settings)
        """
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password

        self._driver: AsyncDriver | None = None
        logger.info(f"GraphManager configured for: {self.uri}")

    async def connect(self) -> None:
        """Stellt die Verbindung zur Neo4j Datenbank her."""
        if self._driver is not None:
            return

        try:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
            )
            # Verbindung testen
            await self._driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j")

        except AuthError:
            logger.error("Neo4j authentication failed")
            raise
        except ServiceUnavailable:
            logger.error(f"Neo4j service unavailable at {self.uri}")
            raise

    async def disconnect(self) -> None:
        """Schließt die Verbindung zur Datenbank."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    @asynccontextmanager
    async def session(self):
        """Context Manager für Neo4j Sessions."""
        if self._driver is None:
            await self.connect()

        session = self._driver.session()
        try:
            yield session
        finally:
            await session.close()

    async def init_schema(self) -> None:
        """
        Initialisiert das Datenbankschema mit Constraints und Indices.

        Erstellt:
        - Uniqueness Constraints für Node IDs
        - Indices für häufige Suchanfragen
        """
        async with self.session() as session:
            # Constraints für Node-Typen
            constraints = [
                "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (n:Person) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (n:Location) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT date_id IF NOT EXISTS FOR (n:Date) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT org_id IF NOT EXISTS FOR (n:Organization) REQUIRE n.id IS UNIQUE",
            ]

            # Indices für Suche
            indices = [
                "CREATE INDEX person_name IF NOT EXISTS FOR (n:Person) ON (n.name)",
                "CREATE INDEX event_name IF NOT EXISTS FOR (n:Event) ON (n.name)",
                "CREATE INDEX location_name IF NOT EXISTS FOR (n:Location) ON (n.name)",
                "CREATE INDEX org_name IF NOT EXISTS FOR (n:Organization) ON (n.name)",
                "CREATE INDEX source_label IF NOT EXISTS FOR (n) ON (n.source_label)",
            ]

            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint may already exist: {e}")

            for index in indices:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.debug(f"Index may already exist: {e}")

            logger.info("Database schema initialized")

    def _node_to_properties(self, node: AnyNode) -> dict[str, Any]:
        """Konvertiert einen Node in Neo4j-kompatible Properties."""
        # Basis-Properties
        props: dict[str, Any] = {
            "id": str(node.id),
            "name": node.name,
            "source_label": node.source_label.value,
            "confidence": node.confidence,
        }

        if node.description:
            props["description"] = node.description
        if node.aliases:
            props["aliases"] = node.aliases

        # Typ-spezifische Properties
        if isinstance(node, PersonNode):
            if node.birth_date:
                props["birth_date"] = node.birth_date.isoformat()
            if node.death_date:
                props["death_date"] = node.death_date.isoformat()
            if node.nationality:
                props["nationality"] = node.nationality
            if node.occupation:
                props["occupation"] = node.occupation

        elif isinstance(node, EventNode):
            if node.start_date:
                props["start_date"] = node.start_date.isoformat()
            if node.end_date:
                props["end_date"] = node.end_date.isoformat()
            if node.event_type:
                props["event_type"] = node.event_type

        elif isinstance(node, LocationNode):
            if node.location_type:
                props["location_type"] = node.location_type
            if node.coordinates:
                props["latitude"] = node.coordinates[0]
                props["longitude"] = node.coordinates[1]
            if node.parent_location:
                props["parent_location"] = node.parent_location

        elif isinstance(node, DateNode):
            props["date_value"] = node.date_value.isoformat()
            props["precision"] = node.precision
            props["calendar_system"] = node.calendar_system

        elif isinstance(node, OrganizationNode):
            if node.org_type:
                props["org_type"] = node.org_type
            if node.founded_date:
                props["founded_date"] = node.founded_date.isoformat()
            if node.dissolved_date:
                props["dissolved_date"] = node.dissolved_date.isoformat()
            if node.headquarters:
                props["headquarters"] = node.headquarters

        return props

    def _get_node_labels(self, node: AnyNode) -> str:
        """
        Generiert die Neo4j Labels für einen Node.

        Kombiniert Typ-Label mit Source-Label (:Claim oder :Fact).
        """
        type_label = node.node_type.value
        source_label = node.source_label.value
        return f":{type_label}:{source_label}"

    async def add_node(self, node: AnyNode, session: AsyncSession | None = None) -> str:
        """
        Fügt einen einzelnen Node zur Datenbank hinzu.

        Verwendet MERGE für Idempotenz (existierende Nodes werden aktualisiert).

        Returns:
            Die ID des erstellten/aktualisierten Nodes
        """
        labels = self._get_node_labels(node)
        props = self._node_to_properties(node)

        query = f"""
        MERGE (n{labels} {{name: $name}})
        SET n += $props
        RETURN n.id as id
        """

        async def _execute(tx):
            result = await tx.run(query, name=node.name, props=props)
            record = await result.single()
            return record["id"] if record else None

        if session:
            return await session.execute_write(_execute)
        else:
            async with self.session() as s:
                return await s.execute_write(_execute)

    async def add_relationship(
        self,
        relationship: Relationship,
        session: AsyncSession | None = None,
    ) -> bool:
        """
        Fügt eine Beziehung zwischen zwei Nodes hinzu.

        Erstellt die Nodes falls sie nicht existieren (mit minimalem Skeleton).

        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        source_label = f":{relationship.source_type.value}"
        target_label = f":{relationship.target_type.value}"
        rel_type = relationship.relation_type.value
        source_marker = f":{relationship.source_label.value}"

        # Properties für die Relationship
        rel_props = {
            "id": str(relationship.id),
            "source_label": relationship.source_label.value,
            "confidence": relationship.confidence,
            **relationship.properties,
        }

        query = f"""
        MERGE (source{source_label}{source_marker} {{name: $source_name}})
        MERGE (target{target_label}{source_marker} {{name: $target_name}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $rel_props
        RETURN r
        """

        async def _execute(tx):
            result = await tx.run(
                query,
                source_name=relationship.source_name,
                target_name=relationship.target_name,
                rel_props=rel_props,
            )
            return await result.single() is not None

        try:
            if session:
                return await session.execute_write(_execute)
            else:
                async with self.session() as s:
                    return await s.execute_write(_execute)
        except Exception as e:
            logger.error(f"Failed to add relationship: {e}")
            return False

    async def add_claim_graph(
        self,
        extraction: KnowledgeGraphExtraction,
    ) -> dict[str, int]:
        """
        Fügt einen extrahierten Knowledge Graph als Claims zur Datenbank hinzu.

        Alle Nodes werden mit dem Label :Claim markiert, um sie von
        verifizierten :Fact Nodes zu unterscheiden.

        Args:
            extraction: Das Extraktionsergebnis mit Nodes und Relationships

        Returns:
            Dictionary mit Statistiken: {"nodes_added": n, "relationships_added": n}
        """
        nodes_added = 0
        relationships_added = 0

        async with self.session() as session:
            # Nodes hinzufügen
            for node in extraction.nodes:
                node.source_label = SourceLabel.CLAIM
                try:
                    await self.add_node(node, session)
                    nodes_added += 1
                except Exception as e:
                    logger.error(f"Failed to add node {node.name}: {e}")

            # Relationships hinzufügen
            for rel in extraction.relationships:
                rel.source_label = SourceLabel.CLAIM
                success = await self.add_relationship(rel, session)
                if success:
                    relationships_added += 1

        logger.info(
            f"Added claim graph: {nodes_added} nodes, {relationships_added} relationships"
        )

        return {
            "nodes_added": nodes_added,
            "relationships_added": relationships_added,
        }

    async def add_fact_graph(
        self,
        extraction: KnowledgeGraphExtraction,
    ) -> dict[str, int]:
        """
        Fügt verifizierte Fakten (Ground Truth) zur Datenbank hinzu.

        Identisch zu add_claim_graph, aber mit :Fact Label.
        """
        nodes_added = 0
        relationships_added = 0

        async with self.session() as session:
            for node in extraction.nodes:
                node.source_label = SourceLabel.FACT
                try:
                    await self.add_node(node, session)
                    nodes_added += 1
                except Exception as e:
                    logger.error(f"Failed to add fact node {node.name}: {e}")

            for rel in extraction.relationships:
                rel.source_label = SourceLabel.FACT
                success = await self.add_relationship(rel, session)
                if success:
                    relationships_added += 1

        logger.info(
            f"Added fact graph: {nodes_added} nodes, {relationships_added} relationships"
        )

        return {
            "nodes_added": nodes_added,
            "relationships_added": relationships_added,
        }

    async def find_matching_facts(
        self,
        claim_name: str,
        claim_type: NodeType,
        similarity_threshold: float = 0.8,
    ) -> list[dict[str, Any]]:
        """
        Sucht nach Fakten, die einer Behauptung entsprechen könnten.

        Verwendet exakte und Fuzzy-Suche auf Namen und Aliasen.
        """
        type_label = claim_type.value

        query = f"""
        MATCH (n:{type_label}:Fact)
        WHERE n.name = $name 
           OR $name IN n.aliases
           OR n.name CONTAINS $name
           OR $name CONTAINS n.name
        RETURN n
        LIMIT 10
        """

        async with self.session() as session:
            result = await session.run(query, name=claim_name)
            records = await result.data()
            return [dict(record["n"]) for record in records]

    async def get_node_relationships(
        self,
        node_name: str,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """
        Holt alle Beziehungen eines Nodes.

        Args:
            node_name: Name des Nodes
            direction: "in", "out", oder "both"
        """
        if direction == "in":
            pattern = "(n)<-[r]-(other)"
        elif direction == "out":
            pattern = "(n)-[r]->(other)"
        else:
            pattern = "(n)-[r]-(other)"

        query = f"""
        MATCH {pattern}
        WHERE n.name = $name
        RETURN type(r) as rel_type, other.name as other_name, labels(other) as other_labels
        """

        async with self.session() as session:
            result = await session.run(query, name=node_name)
            return await result.data()

    async def clear_claims(self) -> int:
        """Löscht alle Claim-Nodes und ihre Beziehungen."""
        query = """
        MATCH (n:Claim)
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        async with self.session() as session:
            result = await session.run(query)
            record = await result.single()
            count = record["deleted"] if record else 0
            logger.info(f"Deleted {count} claim nodes")
            return count

    async def get_statistics(self) -> dict[str, Any]:
        """Gibt Statistiken über den Graph zurück."""
        query = """
        MATCH (n)
        WITH labels(n) as lbls, count(n) as cnt
        UNWIND lbls as label
        RETURN label, sum(cnt) as count
        ORDER BY count DESC
        """
        async with self.session() as session:
            result = await session.run(query)
            records = await result.data()

        stats = {record["label"]: record["count"] for record in records}

        # Relationship-Counts
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        """
        async with self.session() as session:
            result = await session.run(rel_query)
            rel_records = await result.data()

        stats["relationships"] = {
            record["rel_type"]: record["count"] for record in rel_records
        }

        return stats


# Singleton-Pattern für globalen Zugriff
_graph_manager: GraphManager | None = None


async def get_graph_manager() -> GraphManager:
    """Holt oder erstellt den globalen GraphManager."""
    global _graph_manager
    if _graph_manager is None:
        _graph_manager = GraphManager()
        await _graph_manager.connect()
        await _graph_manager.init_schema()
    return _graph_manager
