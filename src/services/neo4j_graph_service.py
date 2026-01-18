"""
Veritas - Neo4j Graph Service

Speichert Mythen, Narrative und deren Beziehungen als Knowledge Graph.
Ermöglicht Graph-basierte Abfragen und Analyse.

Features:
- Mythen als Nodes mit Properties
- Beziehungen zwischen Mythen (RELATED_TO, DEBUNKS, etc.)
- Narrative Patterns als separate Nodes
- Graph-Traversal für verwandte Mythen
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.config import get_settings
from src.models.veritas_schema import (
    HistoricalMyth,
    NarrativePattern,
    MythCategory,
    FactStatus,
    HistoricalEra,
    Region,
)

logger = logging.getLogger(__name__)


class Neo4jGraphService:
    """
    Neo4j Graph Service für Veritas.

    Node Types:
    - Myth: Historische Mythen
    - Narrative: Narrative Patterns
    - Person: Historische Personen
    - Event: Historische Ereignisse
    - Source: Quellen

    Relationship Types:
    - RELATED_TO: Verwandte Mythen
    - BELONGS_TO: Mythos gehört zu Narrativ
    - DEBUNKED_BY: Wurde widerlegt von
    - MENTIONS: Erwähnt Person/Ereignis
    - ORIGINATED_FROM: Ursprung
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        settings = get_settings()

        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password

        self.driver: Optional[Driver] = None
        self._connected = False

    def connect(self) -> bool:
        """Verbindet mit Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
            )
            # Test connection
            self.driver.verify_connectivity()
            self._connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True
        except ServiceUnavailable as e:
            logger.warning(f"Neo4j not available: {e}")
            self._connected = False
            return False
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Neo4j connection error: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Trennt Verbindung."""
        if self.driver:
            self.driver.close()
            self._connected = False
            logger.info("Disconnected from Neo4j")

    @property
    def is_connected(self) -> bool:
        return self._connected and self.driver is not None

    def ensure_connected(self) -> bool:
        """Stellt sicher, dass Verbindung besteht."""
        if not self.is_connected:
            return self.connect()
        return True

    # =========================================================================
    # Schema Setup
    # =========================================================================

    def setup_schema(self):
        """Erstellt Indizes und Constraints."""
        if not self.ensure_connected():
            logger.warning("Cannot setup schema - not connected")
            return

        queries = [
            # Unique constraints
            "CREATE CONSTRAINT myth_id IF NOT EXISTS FOR (m:Myth) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT narrative_id IF NOT EXISTS FOR (n:Narrative) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT source_title IF NOT EXISTS FOR (s:Source) REQUIRE s.title IS UNIQUE",
            # Indexes for faster lookups
            "CREATE INDEX myth_category IF NOT EXISTS FOR (m:Myth) ON (m.category)",
            "CREATE INDEX myth_era IF NOT EXISTS FOR (m:Myth) ON (m.era)",
            "CREATE INDEX myth_status IF NOT EXISTS FOR (m:Myth) ON (m.status)",
            "CREATE INDEX myth_popularity IF NOT EXISTS FOR (m:Myth) ON (m.popularity)",
        ]

        with self.driver.session() as session:
            for query in queries:
                try:
                    session.run(query)
                except Exception as e:
                    # Ignore if already exists
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Schema query failed: {e}")

        logger.info("Neo4j schema setup complete")

    # =========================================================================
    # Import Methods
    # =========================================================================

    def import_myth(self, myth: HistoricalMyth) -> bool:
        """Importiert einen Mythos in den Graph."""
        if not self.ensure_connected():
            return False

        def _get_value(enum_or_str) -> str:
            if isinstance(enum_or_str, str):
                return enum_or_str
            return (
                enum_or_str.value if hasattr(enum_or_str, "value") else str(enum_or_str)
            )

        query = """
        MERGE (m:Myth {id: $id})
        SET m.claim = $claim,
            m.claim_en = $claim_en,
            m.category = $category,
            m.era = $era,
            m.status = $status,
            m.truth = $truth,
            m.truth_en = $truth_en,
            m.popularity = $popularity,
            m.keywords = $keywords,
            m.origin_source = $origin_source,
            m.origin_date = $origin_date,
            m.origin_reason = $origin_reason,
            m.updated_at = datetime()
        RETURN m
        """

        params = {
            "id": myth.id,
            "claim": myth.claim,
            "claim_en": myth.claim_en or "",
            "category": _get_value(myth.category),
            "era": _get_value(myth.era),
            "status": _get_value(myth.status),
            "truth": myth.truth,
            "truth_en": myth.truth_en or "",
            "popularity": myth.popularity,
            "keywords": myth.keywords,
            "origin_source": myth.origin.source if myth.origin else "",
            "origin_date": myth.origin.date if myth.origin else "",
            "origin_reason": myth.origin.reason if myth.origin else "",
        }

        try:
            with self.driver.session() as session:
                session.run(query, params)

                # Create relationships to related myths
                for related_id in myth.related_myths:
                    self._create_relationship(
                        session, myth.id, related_id, "RELATED_TO"
                    )

                # Create debunker relationships
                for debunker in myth.debunked_by:
                    self._create_debunker(session, myth.id, debunker)

                # Create source nodes
                for source in myth.sources:
                    self._create_source(session, myth.id, source)

            logger.debug(f"Imported myth: {myth.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to import myth {myth.id}: {e}")
            return False

    def import_narrative(self, narrative: NarrativePattern) -> bool:
        """Importiert ein Narrativ in den Graph."""
        if not self.ensure_connected():
            return False

        query = """
        MERGE (n:Narrative {id: $id})
        SET n.name = $name,
            n.name_en = $name_en,
            n.description = $description,
            n.typical_claims = $typical_claims,
            n.keywords = $keywords,
            n.purpose = $purpose,
            n.counter_narrative = $counter_narrative,
            n.updated_at = datetime()
        RETURN n
        """

        params = {
            "id": narrative.id,
            "name": narrative.name,
            "name_en": narrative.name_en or "",
            "description": narrative.description,
            "typical_claims": narrative.typical_claims,
            "keywords": narrative.keywords,
            "purpose": narrative.purpose or "",
            "counter_narrative": narrative.counter_narrative or "",
        }

        try:
            with self.driver.session() as session:
                session.run(query, params)

                # Link examples to narrative
                for example_id in narrative.examples:
                    rel_query = """
                    MATCH (m:Myth {id: $myth_id})
                    MATCH (n:Narrative {id: $narrative_id})
                    MERGE (m)-[:BELONGS_TO]->(n)
                    """
                    session.run(
                        rel_query, {"myth_id": example_id, "narrative_id": narrative.id}
                    )

            logger.debug(f"Imported narrative: {narrative.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to import narrative {narrative.id}: {e}")
            return False

    def _create_relationship(self, session, from_id: str, to_id: str, rel_type: str):
        """Erstellt Beziehung zwischen Mythen."""
        query = f"""
        MATCH (m1:Myth {{id: $from_id}})
        MERGE (m2:Myth {{id: $to_id}})
        MERGE (m1)-[:{rel_type}]->(m2)
        """
        session.run(query, {"from_id": from_id, "to_id": to_id})

    def _create_debunker(self, session, myth_id: str, debunker: str):
        """Erstellt Debunker-Node und Beziehung."""
        query = """
        MATCH (m:Myth {id: $myth_id})
        MERGE (p:Person {name: $name})
        MERGE (p)-[:DEBUNKED]->(m)
        """
        session.run(query, {"myth_id": myth_id, "name": debunker})

    def _create_source(self, session, myth_id: str, source):
        """Erstellt Source-Node und Beziehung."""

        def _get_value(enum_or_str) -> str:
            if isinstance(enum_or_str, str):
                return enum_or_str
            return (
                enum_or_str.value if hasattr(enum_or_str, "value") else str(enum_or_str)
            )

        query = """
        MATCH (m:Myth {id: $myth_id})
        MERGE (s:Source {title: $title})
        SET s.type = $type,
            s.author = $author,
            s.year = $year,
            s.url = $url
        MERGE (m)-[:CITED_IN]->(s)
        """
        session.run(
            query,
            {
                "myth_id": myth_id,
                "title": source.title,
                "type": _get_value(source.type),
                "author": source.author or "",
                "year": source.year or 0,
                "url": source.url or "",
            },
        )

    def import_all_from_database(self):
        """Importiert alle Mythen und Narrative aus der myths_database."""
        from src.data.myths_database import get_myths_database

        db = get_myths_database()

        # Setup schema first
        self.setup_schema()

        # Import myths
        imported_myths = 0
        for myth in db.myths.values():
            if self.import_myth(myth):
                imported_myths += 1

        # Import narratives
        imported_narratives = 0
        for narrative in db.narratives.values():
            if self.import_narrative(narrative):
                imported_narratives += 1

        logger.info(
            f"Imported {imported_myths} myths and {imported_narratives} narratives to Neo4j"
        )
        return {
            "myths_imported": imported_myths,
            "narratives_imported": imported_narratives,
        }

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_myth(self, myth_id: str) -> Optional[Dict]:
        """Holt Mythos aus Graph."""
        if not self.ensure_connected():
            return None

        query = """
        MATCH (m:Myth {id: $id})
        RETURN m
        """

        with self.driver.session() as session:
            result = session.run(query, {"id": myth_id})
            record = result.single()
            if record:
                return dict(record["m"])
        return None

    def get_related_myths(self, myth_id: str, depth: int = 2) -> List[Dict]:
        """Findet verwandte Mythen über Graph-Traversal."""
        if not self.ensure_connected():
            return []

        # Build query based on depth (can't use parameter in variable-length pattern)
        if depth == 1:
            query = """
            MATCH (m:Myth {id: $id})-[:RELATED_TO]-(related:Myth)
            WHERE m <> related
            RETURN DISTINCT related, 1 as distance
            ORDER BY related.popularity DESC
            LIMIT 10
            """
        elif depth == 2:
            query = """
            MATCH (m:Myth {id: $id})-[:RELATED_TO*1..2]-(related:Myth)
            WHERE m <> related
            RETURN DISTINCT related, 2 as distance
            ORDER BY related.popularity DESC
            LIMIT 10
            """
        else:
            query = """
            MATCH (m:Myth {id: $id})-[:RELATED_TO*1..3]-(related:Myth)
            WHERE m <> related
            RETURN DISTINCT related, 3 as distance
            ORDER BY related.popularity DESC
            LIMIT 10
            """

        results = []
        try:
            with self.driver.session() as session:
                result = session.run(query, {"id": myth_id})
                for record in result:
                    myth_data = dict(record["related"])
                    myth_data["distance"] = record["distance"]
                    results.append(myth_data)
        except Exception as e:
            logger.error(f"get_related_myths failed: {e}")

        return results

    def get_myths_by_narrative(self, narrative_id: str) -> List[Dict]:
        """Findet alle Mythen eines Narrativs."""
        if not self.ensure_connected():
            return []

        query = """
        MATCH (m:Myth)-[:BELONGS_TO]->(n:Narrative {id: $id})
        RETURN m
        ORDER BY m.popularity DESC
        """

        results = []
        try:
            with self.driver.session() as session:
                result = session.run(query, {"id": narrative_id})
                for record in result:
                    results.append(dict(record["m"]))
        except Exception as e:
            logger.error(f"get_myths_by_narrative failed: {e}")

        return results

    def get_myths_by_debunker(self, debunker_name: str) -> List[Dict]:
        """Findet alle Mythen, die von einer Person widerlegt wurden."""
        if not self.ensure_connected():
            return []

        query = """
        MATCH (p:Person {name: $name})-[:DEBUNKED]->(m:Myth)
        RETURN m
        ORDER BY m.popularity DESC
        """

        results = []
        try:
            with self.driver.session() as session:
                result = session.run(query, {"name": debunker_name})
                for record in result:
                    results.append(dict(record["m"]))
        except Exception as e:
            logger.error(f"get_myths_by_debunker failed: {e}")

        return results

    def search_myths_fulltext(self, query: str, limit: int = 10) -> List[Dict]:
        """Volltextsuche in Mythen."""
        if not self.ensure_connected():
            return []

        search_term = query.lower()

        # Very simple query - just search in claim
        cypher = """
        MATCH (m:Myth)
        WHERE toLower(m.claim) CONTAINS $query 
        RETURN m
        ORDER BY m.popularity DESC
        LIMIT 10
        """

        results = []
        try:
            with self.driver.session() as session:
                result = session.run(cypher, {"query": search_term})
                for record in result:
                    node = record["m"]
                    results.append(
                        {
                            "id": node.get("id"),
                            "claim": node.get("claim"),
                            "status": node.get("status"),
                            "popularity": node.get("popularity"),
                        }
                    )
        except Exception as e:
            logger.error(f"search_myths_fulltext failed: {e}")

        return results

    def get_graph_stats(self) -> Dict:
        """Gibt Graph-Statistiken zurück."""
        if not self.ensure_connected():
            return {"connected": False}

        stats = {"connected": True}

        queries = {
            "total_myths": "MATCH (m:Myth) RETURN count(m) as count",
            "total_narratives": "MATCH (n:Narrative) RETURN count(n) as count",
            "total_persons": "MATCH (p:Person) RETURN count(p) as count",
            "total_sources": "MATCH (s:Source) RETURN count(s) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
        }

        with self.driver.session() as session:
            for key, query in queries.items():
                result = session.run(query)
                record = result.single()
                stats[key] = record["count"] if record else 0

        return stats

    def get_category_distribution(self) -> Dict[str, int]:
        """Gibt Verteilung nach Kategorien zurück."""
        if not self.ensure_connected():
            return {}

        query = """
        MATCH (m:Myth)
        RETURN m.category as category, count(*) as count
        ORDER BY count DESC
        """

        distribution = {}
        try:
            with self.driver.session() as session:
                result = session.run(query)
                for record in result:
                    if record["category"]:
                        distribution[record["category"]] = record["count"]
        except Exception as e:
            logger.error(f"get_category_distribution failed: {e}")

        return distribution

    def find_myth_clusters(self) -> List[Dict]:
        """Findet Cluster verwandter Mythen."""
        if not self.ensure_connected():
            return []

        # Very simple - just get myths with most debunkers as proxy for "importance"
        query = """
        MATCH (p:Person)-[:DEBUNKED]->(m:Myth)
        WITH m.id as myth_id, m.claim as claim, count(p) as connections
        RETURN myth_id, claim, connections
        ORDER BY connections DESC
        LIMIT 20
        """

        clusters = []
        try:
            with self.driver.session() as session:
                result = session.run(query)
                for record in result:
                    clusters.append(
                        {
                            "myth_id": record["myth_id"],
                            "claim": record["claim"],
                            "related_ids": [],
                            "connections": record["connections"],
                        }
                    )
        except Exception as e:
            logger.error(f"find_myth_clusters failed: {e}")

        return clusters

    def clear_database(self):
        """Löscht alle Daten (Vorsicht!)."""
        if not self.ensure_connected():
            return

        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

        logger.warning("Neo4j database cleared!")


# =============================================================================
# Singleton
# =============================================================================

_graph_service: Optional[Neo4jGraphService] = None


def get_graph_service() -> Neo4jGraphService:
    """Gibt Graph Service Instanz zurück."""
    global _graph_service
    if _graph_service is None:
        _graph_service = Neo4jGraphService()
    return _graph_service
