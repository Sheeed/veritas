"""
Verbesserter Extraction Agent mit professionellen AI-Techniken.

Features:
- Self-Consistency: Mehrfache Extraktion und Konsens-Bildung
- Chain-of-Thought: Strukturiertes Reasoning
- Confidence Calibration: Realistische Konfidenzwerte
- Anti-Hallucination: Strikte Quellenprüfung
- Multi-Provider: OpenAI, Groq (kostenlos), Mistral
"""

import asyncio
import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from src.config import get_settings
from src.models.schema import (
    KnowledgeGraphExtraction,
    SourceLabel,
)
from src.agents.prompts_v2 import PromptBuilder
from src.llm.client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


class ExtractionAgentV2:
    """
    Verbesserter Extraction Agent mit Self-Consistency.
    
    Unterstuetzt mehrere LLM Provider:
    - OpenAI (GPT-4o)
    - Groq (Llama 3.3 70B) - KOSTENLOS
    - Mistral
    
    Fuehrt mehrfache Extraktionen durch und bildet einen Konsens
    fuer robustere Ergebnisse.
    """
    
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        num_extractions: int = 1,  # Fuer Self-Consistency auf 3 setzen
        consensus_threshold: float = 0.6,
    ):
        settings = get_settings()
        
        # Verwende konfigurierten Provider wenn nicht angegeben
        self.provider = provider or settings.llm_provider
        self.model = model or settings.get_active_model()
        
        # LLM Client initialisieren
        self.client = get_llm_client(provider=self.provider, model=self.model)
        
        self.temperature = temperature
        self.num_extractions = num_extractions
        self.consensus_threshold = consensus_threshold
        
        logger.info(f"ExtractionAgent initialisiert: {self.provider} / {self.model}")
    
    async def extract_knowledge_graph(
        self,
        text: str,
        use_cot: bool = True,
        use_few_shot: bool = True,
        mark_as_claim: bool = True,
    ) -> KnowledgeGraphExtraction:
        """
        Extrahiert einen Knowledge Graph aus Text.
        
        Args:
            text: Eingabetext
            use_cot: Chain-of-Thought Reasoning aktivieren
            use_few_shot: Few-Shot Beispiele verwenden
            mark_as_claim: Als unverifizierter Claim markieren
        
        Returns:
            KnowledgeGraphExtraction mit extrahierten Entitäten
        """
        if self.num_extractions > 1:
            return await self._extract_with_consistency(
                text, use_cot, use_few_shot, mark_as_claim
            )
        else:
            return await self._single_extraction(
                text, use_cot, use_few_shot, mark_as_claim
            )
    
    async def _single_extraction(
        self,
        text: str,
        use_cot: bool,
        use_few_shot: bool,
        mark_as_claim: bool,
    ) -> KnowledgeGraphExtraction:
        """Fuehrt eine einzelne Extraktion durch."""
        
        messages = PromptBuilder.build_extraction_prompt(
            text=text,
            use_cot=use_cot,
            use_few_shot=use_few_shot,
        )
        
        # Nutze den abstrahierten LLM Client
        content = await self.client.chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        
        extraction = self._parse_response(content, text)
        
        # Als Claim markieren
        if mark_as_claim:
            for node in extraction.nodes:
                node.source_label = SourceLabel.CLAIM
        
        return extraction
    
    async def _extract_with_consistency(
        self,
        text: str,
        use_cot: bool,
        use_few_shot: bool,
        mark_as_claim: bool,
    ) -> KnowledgeGraphExtraction:
        """
        Führt mehrfache Extraktionen durch und bildet Konsens.
        
        Self-Consistency verbessert die Zuverlässigkeit durch:
        1. Mehrfache unabhängige Extraktionen
        2. Identifikation übereinstimmender Fakten
        3. Ablehnung von Einzelmeinungen (Halluzinationen)
        """
        
        # Führe N Extraktionen parallel durch
        tasks = []
        for i in range(self.num_extractions):
            # Leicht verschiedene Temperaturen für Diversität
            temp = self.temperature + (i * 0.1)
            tasks.append(self._single_extraction_with_temp(
                text, use_cot, use_few_shot, temp
            ))
        
        extractions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtere erfolgreiche Extraktionen
        valid_extractions = [
            e for e in extractions 
            if isinstance(e, KnowledgeGraphExtraction)
        ]
        
        if not valid_extractions:
            raise ValueError("All extractions failed")
        
        if len(valid_extractions) == 1:
            return valid_extractions[0]
        
        # Konsens bilden
        consensus = self._build_consensus(valid_extractions)
        
        if mark_as_claim:
            for node in consensus.nodes:
                node.source_label = SourceLabel.CLAIM
        
        return consensus
    
    async def _single_extraction_with_temp(
        self,
        text: str,
        use_cot: bool,
        use_few_shot: bool,
        temperature: float,
    ) -> KnowledgeGraphExtraction:
        """Einzelextraktion mit spezifischer Temperatur."""
        
        messages = PromptBuilder.build_extraction_prompt(
            text=text,
            use_cot=use_cot,
            use_few_shot=use_few_shot,
        )
        
        content = await self.client.chat_completion(
            messages=messages,
            temperature=min(temperature, 1.0),
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        
        return self._parse_response(content, text)
    
    def _build_consensus(
        self,
        extractions: list[KnowledgeGraphExtraction],
    ) -> KnowledgeGraphExtraction:
        """
        Bildet Konsens aus mehreren Extraktionen.
        
        Ein Fakt wird nur übernommen, wenn er in mindestens
        `consensus_threshold` der Extraktionen vorkommt.
        """
        
        # Zähle Node-Vorkommen
        node_counts: dict[str, int] = {}
        node_data: dict[str, Any] = {}
        
        for extraction in extractions:
            for node in extraction.nodes:
                key = f"{node.node_type.value}:{node.name.lower()}"
                node_counts[key] = node_counts.get(key, 0) + 1
                
                # Behalte Node mit höchster Konfidenz
                if key not in node_data or node.confidence > node_data[key].confidence:
                    node_data[key] = node
        
        # Zähle Relationship-Vorkommen
        rel_counts: dict[str, int] = {}
        rel_data: dict[str, Any] = {}
        
        for extraction in extractions:
            for rel in extraction.relationships:
                key = f"{rel.source_name.lower()}:{rel.relation_type.value}:{rel.target_name.lower()}"
                rel_counts[key] = rel_counts.get(key, 0) + 1
                
                if key not in rel_data or rel.confidence > rel_data[key].confidence:
                    rel_data[key] = rel
        
        # Filtere nach Konsens-Threshold
        num_extractions = len(extractions)
        min_count = int(num_extractions * self.consensus_threshold)
        
        consensus_nodes = [
            node for key, node in node_data.items()
            if node_counts[key] >= min_count
        ]
        
        consensus_relationships = [
            rel for key, rel in rel_data.items()
            if rel_counts[key] >= min_count
        ]
        
        # Passe Konfidenz basierend auf Konsens an
        for node in consensus_nodes:
            key = f"{node.node_type.value}:{node.name.lower()}"
            consensus_ratio = node_counts[key] / num_extractions
            # Konfidenz wird durch Konsens verstärkt oder abgeschwächt
            node.confidence = node.confidence * (0.5 + 0.5 * consensus_ratio)
        
        return KnowledgeGraphExtraction(
            nodes=consensus_nodes,
            relationships=consensus_relationships,
            extraction_metadata={
                "method": "self_consistency",
                "num_extractions": num_extractions,
                "consensus_threshold": self.consensus_threshold,
            },
        )
    
    def _parse_response(
        self,
        content: str,
        source_text: str,
    ) -> KnowledgeGraphExtraction:
        """Parst die LLM-Response zu einer Extraktion."""
        
        # Entferne CoT-Reasoning vor dem JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            content = json_match.group()
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            # Versuche Reparatur
            content = self._repair_json(content)
            data = json.loads(content)
        
        # Extrahiere Nodes und Relationships
        nodes_data = data.get("nodes", data.get("entities", []))
        rels_data = data.get("relationships", data.get("relations", []))
        
        # Validiere und konvertiere
        try:
            extraction = KnowledgeGraphExtraction(
                nodes=nodes_data,
                relationships=rels_data,
                source_text_hash=self._hash_text(source_text),
                extraction_metadata=data.get("metadata", {}),
            )
            return extraction
        except ValidationError as e:
            logger.warning(f"Validation error, attempting repair: {e}")
            # Entferne ungültige Einträge
            valid_nodes = self._filter_valid_nodes(nodes_data)
            valid_rels = self._filter_valid_relationships(rels_data)
            
            return KnowledgeGraphExtraction(
                nodes=valid_nodes,
                relationships=valid_rels,
                source_text_hash=self._hash_text(source_text),
            )
    
    def _repair_json(self, content: str) -> str:
        """Versucht beschädigtes JSON zu reparieren."""
        # Entferne trailing commas
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        
        # Füge fehlende Quotes hinzu
        content = re.sub(r'(\w+):', r'"\1":', content)
        
        return content
    
    def _filter_valid_nodes(self, nodes: list) -> list:
        """Filtert ungültige Nodes."""
        valid = []
        for node in nodes:
            if isinstance(node, dict) and node.get("name") and node.get("node_type"):
                # Normalisiere node_type
                node_type = node["node_type"]
                if isinstance(node_type, str):
                    node["node_type"] = node_type.capitalize()
                valid.append(node)
        return valid
    
    def _filter_valid_relationships(self, rels: list) -> list:
        """Filtert ungültige Relationships."""
        valid = []
        for rel in rels:
            if isinstance(rel, dict):
                if rel.get("source_name") and rel.get("target_name") and rel.get("relation_type"):
                    valid.append(rel)
        return valid
    
    def _hash_text(self, text: str) -> str:
        """Erzeugt einen Hash für den Quelltext."""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    async def decompose_facts(self, text: str) -> list[dict[str, str]]:
        """
        Zerlegt einen Text in atomare Fakten.
        
        Nuetzlich fuer granulare Verifikation.
        """
        messages = PromptBuilder.build_decomposition_prompt(text)
        
        content = await self.client.chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            return data.get("facts", [])
        except json.JSONDecodeError:
            return []
    
    async def check_consistency(
        self,
        extraction1: KnowledgeGraphExtraction,
        extraction2: KnowledgeGraphExtraction,
    ) -> dict[str, Any]:
        """
        Prueft die Konsistenz zweier Extraktionen.
        
        Nuetzlich fuer Quality Assurance.
        """
        messages = PromptBuilder.build_consistency_prompt(
            extraction1.model_dump(),
            extraction2.model_dump(),
        )
        
        content = await self.client.chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Failed to parse consistency check"}


# =============================================================================
# Evaluation Framework
# =============================================================================


class ExtractionEvaluator:
    """
    Evaluiert die Qualität von Extraktionen.
    
    Metriken:
    - Precision: Anteil korrekter extrahierter Fakten
    - Recall: Anteil gefundener wahrer Fakten
    - F1: Harmonisches Mittel
    - Hallucination Rate: Anteil erfundener Fakten
    """
    
    @staticmethod
    def calculate_metrics(
        predicted: KnowledgeGraphExtraction,
        ground_truth: KnowledgeGraphExtraction,
    ) -> dict[str, float]:
        """Berechnet Evaluationsmetriken."""
        
        # Node-Metriken
        pred_nodes = {f"{n.node_type.value}:{n.name.lower()}" for n in predicted.nodes}
        true_nodes = {f"{n.node_type.value}:{n.name.lower()}" for n in ground_truth.nodes}
        
        node_tp = len(pred_nodes & true_nodes)
        node_fp = len(pred_nodes - true_nodes)
        node_fn = len(true_nodes - pred_nodes)
        
        node_precision = node_tp / (node_tp + node_fp) if (node_tp + node_fp) > 0 else 0
        node_recall = node_tp / (node_tp + node_fn) if (node_tp + node_fn) > 0 else 0
        node_f1 = 2 * node_precision * node_recall / (node_precision + node_recall) if (node_precision + node_recall) > 0 else 0
        
        # Relationship-Metriken
        pred_rels = {
            f"{r.source_name.lower()}:{r.relation_type.value}:{r.target_name.lower()}"
            for r in predicted.relationships
        }
        true_rels = {
            f"{r.source_name.lower()}:{r.relation_type.value}:{r.target_name.lower()}"
            for r in ground_truth.relationships
        }
        
        rel_tp = len(pred_rels & true_rels)
        rel_fp = len(pred_rels - true_rels)
        rel_fn = len(true_rels - pred_rels)
        
        rel_precision = rel_tp / (rel_tp + rel_fp) if (rel_tp + rel_fp) > 0 else 0
        rel_recall = rel_tp / (rel_tp + rel_fn) if (rel_tp + rel_fn) > 0 else 0
        rel_f1 = 2 * rel_precision * rel_recall / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0
        
        # Hallucination Rate
        hallucination_rate = node_fp / len(pred_nodes) if len(pred_nodes) > 0 else 0
        
        return {
            "node_precision": node_precision,
            "node_recall": node_recall,
            "node_f1": node_f1,
            "relationship_precision": rel_precision,
            "relationship_recall": rel_recall,
            "relationship_f1": rel_f1,
            "hallucination_rate": hallucination_rate,
            "overall_f1": (node_f1 + rel_f1) / 2,
        }
