"""
Knowledge Graph Extraction Agent.

Verwendet OpenAI GPT-4o via LlamaIndex für strukturierte Extraktion
von Entitäten und Beziehungen aus Text.
"""

import hashlib
import logging
from typing import Any

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from pydantic import ValidationError

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
    RelationType,
    SourceLabel,
)

from .prompts import EXTRACTION_SYSTEM_PROMPT, get_few_shot_examples

logger = logging.getLogger(__name__)


class ExtractionAgent:
    """
    Agent zur Extraktion von Knowledge Graphs aus Text.

    Verwendet Structured Output via OpenAI Function Calling,
    um konsistente, validierte Pydantic-Modelle zu erzeugen.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialisiert den Extraction Agent.

        Args:
            model: OpenAI Model (default: aus Settings)
            temperature: LLM Temperature (0.0 für deterministische Ausgaben)
            max_retries: Anzahl Wiederholungen bei Parsing-Fehlern
        """
        settings = get_settings()
        self.model_name = model or settings.openai_model
        self.temperature = temperature
        self.max_retries = max_retries

        self.llm = OpenAI(
            model=self.model_name,
            api_key=settings.openai_api_key,
            temperature=temperature,
        )

        logger.info(f"ExtractionAgent initialized with model: {self.model_name}")

    def _build_messages(
        self, text: str, use_few_shot: bool = True
    ) -> list[ChatMessage]:
        """Erstellt die Message-Liste für den LLM-Aufruf."""
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=EXTRACTION_SYSTEM_PROMPT)
        ]

        # Few-Shot Beispiele für bessere Ergebnisse
        if use_few_shot:
            for example in get_few_shot_examples():
                role = (
                    MessageRole.USER
                    if example["role"] == "user"
                    else MessageRole.ASSISTANT
                )
                messages.append(ChatMessage(role=role, content=example["content"]))

        # Eigentliche Anfrage
        user_prompt = f"""Extrahiere den Knowledge Graph aus folgendem Text.
Antworte NUR mit validem JSON im Format des KnowledgeGraphExtraction Schemas.

TEXT:
\"\"\"
{text}
\"\"\"

Extrahierter Knowledge Graph (JSON):"""

        messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))
        return messages

    def _parse_node(self, node_data: dict[str, Any]) -> AnyNode | None:
        """Parst ein Node-Dictionary in das entsprechende Pydantic-Modell."""
        try:
            node_type_str = node_data.get("node_type", "")

            # Mapping zu den richtigen Modellklassen
            type_mapping = {
                "Person": PersonNode,
                NodeType.PERSON.value: PersonNode,
                NodeType.PERSON: PersonNode,
                "Event": EventNode,
                NodeType.EVENT.value: EventNode,
                NodeType.EVENT: EventNode,
                "Location": LocationNode,
                NodeType.LOCATION.value: LocationNode,
                NodeType.LOCATION: LocationNode,
                "Date": DateNode,
                NodeType.DATE.value: DateNode,
                NodeType.DATE: DateNode,
                "Organization": OrganizationNode,
                NodeType.ORGANIZATION.value: OrganizationNode,
                NodeType.ORGANIZATION: OrganizationNode,
            }

            model_class = type_mapping.get(node_type_str)
            if model_class is None:
                logger.warning(f"Unknown node type: {node_type_str}")
                return None

            # Setze den korrekten NodeType Enum-Wert
            if isinstance(node_type_str, str):
                node_data["node_type"] = NodeType(node_type_str)

            return model_class.model_validate(node_data)

        except ValidationError as e:
            logger.warning(f"Failed to parse node: {e}")
            return None

    def _parse_relationship(self, rel_data: dict[str, Any]) -> Relationship | None:
        """Parst ein Relationship-Dictionary in das Pydantic-Modell."""
        try:
            # Konvertiere String-Werte zu Enums
            if isinstance(rel_data.get("source_type"), str):
                rel_data["source_type"] = NodeType(rel_data["source_type"])
            if isinstance(rel_data.get("target_type"), str):
                rel_data["target_type"] = NodeType(rel_data["target_type"])
            if isinstance(rel_data.get("relation_type"), str):
                rel_data["relation_type"] = RelationType(rel_data["relation_type"])
            if isinstance(rel_data.get("source_label"), str):
                rel_data["source_label"] = SourceLabel(rel_data["source_label"])

            return Relationship.model_validate(rel_data)

        except (ValidationError, ValueError) as e:
            logger.warning(f"Failed to parse relationship: {e}")
            return None

    def _parse_llm_response(
        self, response_text: str, source_text: str
    ) -> KnowledgeGraphExtraction:
        """
        Parst die LLM-Antwort in ein KnowledgeGraphExtraction-Objekt.

        Robuste Behandlung von JSON-Parsing-Fehlern und ungültigen Daten.
        """
        import json

        # JSON aus der Antwort extrahieren
        response_text = response_text.strip()

        # Entferne mögliche Markdown-Code-Blöcke
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.debug(f"Raw response: {response_text[:500]}...")
            return KnowledgeGraphExtraction(
                source_text_hash=hashlib.sha256(source_text.encode()).hexdigest()
            )

        # Nodes parsen
        nodes: list[AnyNode] = []
        for node_data in data.get("nodes", []):
            parsed_node = self._parse_node(node_data)
            if parsed_node is not None:
                nodes.append(parsed_node)

        # Relationships parsen
        relationships: list[Relationship] = []
        for rel_data in data.get("relationships", []):
            parsed_rel = self._parse_relationship(rel_data)
            if parsed_rel is not None:
                relationships.append(parsed_rel)

        return KnowledgeGraphExtraction(
            nodes=nodes,
            relationships=relationships,
            source_text_hash=hashlib.sha256(source_text.encode()).hexdigest(),
            extraction_metadata={
                "model": self.model_name,
                "temperature": self.temperature,
                "node_count": len(nodes),
                "relationship_count": len(relationships),
            },
        )

    async def extract_knowledge_graph(
        self,
        text: str,
        use_few_shot: bool = True,
        mark_as_claim: bool = True,
    ) -> KnowledgeGraphExtraction:
        """
        Extrahiert einen Knowledge Graph aus dem gegebenen Text.

        Args:
            text: Der zu analysierende Text
            use_few_shot: Ob Few-Shot Beispiele verwendet werden sollen
            mark_as_claim: Ob extrahierte Daten als :Claim markiert werden sollen

        Returns:
            KnowledgeGraphExtraction mit Nodes und Relationships
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for extraction")
            return KnowledgeGraphExtraction()

        messages = self._build_messages(text, use_few_shot)

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Extraction attempt {attempt + 1}/{self.max_retries}")

                # LLM-Aufruf (async)
                response = await self.llm.achat(messages)
                response_text = response.message.content

                if not response_text:
                    logger.warning("Empty response from LLM")
                    continue

                # Parsen der Antwort
                extraction = self._parse_llm_response(response_text, text)

                # Validierung: Mindestens ein Node oder Relationship
                if extraction.nodes or extraction.relationships:
                    # Markiere als Claim wenn gewünscht
                    if mark_as_claim:
                        for node in extraction.nodes:
                            node.source_label = SourceLabel.CLAIM
                        for rel in extraction.relationships:
                            rel.source_label = SourceLabel.CLAIM

                    logger.info(
                        f"Extraction successful: {len(extraction.nodes)} nodes, "
                        f"{len(extraction.relationships)} relationships"
                    )
                    return extraction

                logger.warning("Extraction returned no nodes or relationships")

            except Exception as e:
                logger.error(f"Extraction attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise

        # Fallback: Leeres Ergebnis
        return KnowledgeGraphExtraction(
            source_text_hash=hashlib.sha256(text.encode()).hexdigest()
        )

    def extract_knowledge_graph_sync(
        self,
        text: str,
        use_few_shot: bool = True,
        mark_as_claim: bool = True,
    ) -> KnowledgeGraphExtraction:
        """
        Synchrone Version von extract_knowledge_graph.

        Für Verwendung außerhalb von async Kontexten.
        """
        import asyncio

        return asyncio.run(
            self.extract_knowledge_graph(text, use_few_shot, mark_as_claim)
        )


# Convenience Function
async def extract_knowledge_graph(
    text: str,
    model: str | None = None,
    use_few_shot: bool = True,
) -> KnowledgeGraphExtraction:
    """
    Convenience Function für schnelle Extraktion.

    Erstellt temporär einen Agent und führt die Extraktion durch.
    """
    agent = ExtractionAgent(model=model)
    return await agent.extract_knowledge_graph(text, use_few_shot)
