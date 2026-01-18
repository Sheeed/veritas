"""
Batch Processing Module für History Guardian.

Ermöglicht die parallele Verarbeitung großer Textmengen mit:
- Async Task Queue
- Progress Tracking
- Rate Limiting
- Fehlerbehandlung und Retry-Logik
"""

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.agents.extraction import ExtractionAgent
from src.db.graph_db import GraphManager
from src.models.schema import KnowledgeGraphExtraction

logger = logging.getLogger(__name__)


class BatchStatus(str, Enum):
    """Status eines Batch-Jobs."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ItemStatus(str, Enum):
    """Status eines einzelnen Items im Batch."""

    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchItem:
    """Ein einzelnes Item in einem Batch-Job."""

    id: UUID = field(default_factory=uuid4)
    text: str = ""
    source_id: str | None = None
    status: ItemStatus = ItemStatus.QUEUED
    extraction: KnowledgeGraphExtraction | None = None
    error_message: str | None = None
    processing_time_ms: float = 0.0
    retries: int = 0

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


class BatchJobConfig(BaseModel):
    """Konfiguration für einen Batch-Job."""

    max_concurrent: int = Field(
        default=5, ge=1, le=20, description="Max parallele Verarbeitungen"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Max Wiederholungen bei Fehler"
    )
    retry_delay_seconds: float = Field(
        default=1.0, ge=0.1, description="Wartezeit zwischen Retries"
    )
    timeout_seconds: float = Field(
        default=60.0, ge=10.0, description="Timeout pro Item"
    )
    rate_limit_per_minute: int = Field(
        default=60, ge=1, description="Max Requests pro Minute"
    )
    skip_duplicates: bool = Field(default=True, description="Duplikate überspringen")
    store_results: bool = Field(default=True, description="Ergebnisse in DB speichern")
    as_facts: bool = Field(
        default=False, description="Als Facts statt Claims speichern"
    )


class BatchJobProgress(BaseModel):
    """Fortschrittsinformationen eines Batch-Jobs."""

    job_id: UUID
    status: BatchStatus
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    skipped_items: int
    progress_percent: float
    estimated_remaining_seconds: float | None
    started_at: datetime | None
    completed_at: datetime | None
    current_rate_per_minute: float


class BatchJobResult(BaseModel):
    """Endergebnis eines Batch-Jobs."""

    job_id: UUID
    status: BatchStatus
    config: BatchJobConfig
    progress: BatchJobProgress
    total_nodes_extracted: int
    total_relationships_extracted: int
    total_processing_time_seconds: float
    failed_items: list[dict[str, Any]]


@dataclass
class BatchJob:
    """Ein Batch-Verarbeitungsjob."""

    id: UUID = field(default_factory=uuid4)
    config: BatchJobConfig = field(default_factory=BatchJobConfig)
    items: list[BatchItem] = field(default_factory=list)
    status: BatchStatus = BatchStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    _processed_hashes: set[str] = field(default_factory=set)

    def add_text(self, text: str, source_id: str | None = None) -> BatchItem:
        """Fügt einen Text zum Batch hinzu."""
        item = BatchItem(text=text, source_id=source_id)
        self.items.append(item)
        return item

    def add_texts(self, texts: list[str]) -> list[BatchItem]:
        """Fügt mehrere Texte zum Batch hinzu."""
        return [self.add_text(text) for text in texts]

    @property
    def progress(self) -> BatchJobProgress:
        """Berechnet den aktuellen Fortschritt."""
        total = len(self.items)
        processed = sum(
            1
            for i in self.items
            if i.status in [ItemStatus.SUCCESS, ItemStatus.FAILED, ItemStatus.SKIPPED]
        )
        successful = sum(1 for i in self.items if i.status == ItemStatus.SUCCESS)
        failed = sum(1 for i in self.items if i.status == ItemStatus.FAILED)
        skipped = sum(1 for i in self.items if i.status == ItemStatus.SKIPPED)

        progress_percent = (processed / total * 100) if total > 0 else 0

        # Geschätzte Restzeit
        estimated_remaining = None
        if self.started_at and processed > 0:
            elapsed = (datetime.now() - self.started_at).total_seconds()
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = total - processed
            estimated_remaining = remaining / rate if rate > 0 else None

        # Aktuelle Rate
        current_rate = 0.0
        if self.started_at:
            elapsed_minutes = (datetime.now() - self.started_at).total_seconds() / 60
            current_rate = processed / elapsed_minutes if elapsed_minutes > 0 else 0

        return BatchJobProgress(
            job_id=self.id,
            status=self.status,
            total_items=total,
            processed_items=processed,
            successful_items=successful,
            failed_items=failed,
            skipped_items=skipped,
            progress_percent=progress_percent,
            estimated_remaining_seconds=estimated_remaining,
            started_at=self.started_at,
            completed_at=self.completed_at,
            current_rate_per_minute=current_rate,
        )


class BatchProcessor:
    """
    Async Batch Processor für Knowledge Graph Extraktion.

    Features:
    - Parallele Verarbeitung mit konfigurierbarer Concurrency
    - Rate Limiting zur API-Schonung
    - Automatische Retries bei transienten Fehlern
    - Duplikaterkennung
    - Progress Streaming via AsyncIterator
    """

    def __init__(
        self,
        extraction_agent: ExtractionAgent | None = None,
        graph_manager: GraphManager | None = None,
    ):
        self.extraction_agent = extraction_agent or ExtractionAgent()
        self.graph_manager = graph_manager
        self._jobs: dict[UUID, BatchJob] = {}
        self._rate_limiter: asyncio.Semaphore | None = None
        self._rate_limit_window: list[datetime] = []

    async def _check_rate_limit(self, limit_per_minute: int) -> None:
        """Enforces rate limiting."""
        now = datetime.now()
        # Entferne alte Einträge (älter als 1 Minute)
        self._rate_limit_window = [
            t for t in self._rate_limit_window if (now - t).total_seconds() < 60
        ]

        if len(self._rate_limit_window) >= limit_per_minute:
            # Warte bis ältester Eintrag > 1 Minute alt ist
            oldest = min(self._rate_limit_window)
            wait_time = 60 - (now - oldest).total_seconds()
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        self._rate_limit_window.append(now)

    async def _process_item(
        self,
        item: BatchItem,
        config: BatchJobConfig,
        processed_hashes: set[str],
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Verarbeitet ein einzelnes Item."""
        async with semaphore:
            # Duplikat-Check
            if config.skip_duplicates and item.text_hash in processed_hashes:
                item.status = ItemStatus.SKIPPED
                logger.debug(f"Skipping duplicate: {item.text_hash}")
                return

            processed_hashes.add(item.text_hash)
            item.status = ItemStatus.PROCESSING
            start_time = datetime.now()

            for attempt in range(config.max_retries + 1):
                try:
                    # Rate Limiting
                    await self._check_rate_limit(config.rate_limit_per_minute)

                    # Extraktion mit Timeout
                    extraction = await asyncio.wait_for(
                        self.extraction_agent.extract_knowledge_graph(
                            text=item.text,
                            mark_as_claim=not config.as_facts,
                        ),
                        timeout=config.timeout_seconds,
                    )

                    item.extraction = extraction
                    item.status = ItemStatus.SUCCESS

                    # Optional: In DB speichern
                    if config.store_results and self.graph_manager:
                        if config.as_facts:
                            await self.graph_manager.add_fact_graph(extraction)
                        else:
                            await self.graph_manager.add_claim_graph(extraction)

                    break

                except asyncio.TimeoutError:
                    item.retries = attempt + 1
                    item.error_message = f"Timeout after {config.timeout_seconds}s"
                    logger.warning(f"Item {item.id} timed out (attempt {attempt + 1})")

                except Exception as e:
                    item.retries = attempt + 1
                    item.error_message = str(e)
                    logger.warning(
                        f"Item {item.id} failed (attempt {attempt + 1}): {e}"
                    )

                if attempt < config.max_retries:
                    await asyncio.sleep(config.retry_delay_seconds * (attempt + 1))

            if item.status != ItemStatus.SUCCESS:
                item.status = ItemStatus.FAILED

            item.processing_time_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000

    def create_job(self, config: BatchJobConfig | None = None) -> BatchJob:
        """Erstellt einen neuen Batch-Job."""
        job = BatchJob(config=config or BatchJobConfig())
        self._jobs[job.id] = job
        return job

    def get_job(self, job_id: UUID) -> BatchJob | None:
        """Holt einen Job nach ID."""
        return self._jobs.get(job_id)

    async def process_job(self, job: BatchJob) -> BatchJobResult:
        """
        Verarbeitet einen kompletten Batch-Job.

        Returns:
            BatchJobResult mit allen Statistiken
        """
        if not job.items:
            raise ValueError("Job has no items to process")

        job.status = BatchStatus.PROCESSING
        job.started_at = datetime.now()

        semaphore = asyncio.Semaphore(job.config.max_concurrent)
        processed_hashes: set[str] = set()

        try:
            # Alle Items parallel verarbeiten
            tasks = [
                self._process_item(item, job.config, processed_hashes, semaphore)
                for item in job.items
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            job.status = BatchStatus.COMPLETED

        except Exception as e:
            logger.error(f"Batch job {job.id} failed: {e}")
            job.status = BatchStatus.FAILED

        finally:
            job.completed_at = datetime.now()

        # Statistiken berechnen
        total_nodes = sum(len(i.extraction.nodes) for i in job.items if i.extraction)
        total_rels = sum(
            len(i.extraction.relationships) for i in job.items if i.extraction
        )
        total_time = (job.completed_at - job.started_at).total_seconds()

        failed_items = [
            {"id": str(i.id), "error": i.error_message, "source_id": i.source_id}
            for i in job.items
            if i.status == ItemStatus.FAILED
        ]

        return BatchJobResult(
            job_id=job.id,
            status=job.status,
            config=job.config,
            progress=job.progress,
            total_nodes_extracted=total_nodes,
            total_relationships_extracted=total_rels,
            total_processing_time_seconds=total_time,
            failed_items=failed_items,
        )

    async def process_job_with_progress(
        self,
        job: BatchJob,
        progress_interval: float = 1.0,
    ) -> AsyncIterator[BatchJobProgress]:
        """
        Verarbeitet einen Job und yielded regelmäßig Progress-Updates.

        Usage:
            async for progress in processor.process_job_with_progress(job):
                print(f"Progress: {progress.progress_percent:.1f}%")
        """
        # Starte Verarbeitung im Hintergrund
        process_task = asyncio.create_task(self.process_job(job))

        while not process_task.done():
            yield job.progress
            await asyncio.sleep(progress_interval)

        # Finaler Status
        yield job.progress

        # Propagiere Exceptions
        await process_task

    async def cancel_job(self, job_id: UUID) -> bool:
        """Bricht einen laufenden Job ab."""
        job = self._jobs.get(job_id)
        if job and job.status == BatchStatus.PROCESSING:
            job.status = BatchStatus.CANCELLED
            job.completed_at = datetime.now()
            return True
        return False


class BatchFileProcessor:
    """
    Verarbeitet Dateien im Batch-Modus.

    Unterstützte Formate:
    - TXT (ein Text pro Datei)
    - CSV (eine Spalte mit Texten)
    - JSON (Array von Texten oder Objekten mit 'text' Feld)
    - JSONL (ein JSON-Objekt pro Zeile)
    """

    def __init__(self, processor: BatchProcessor):
        self.processor = processor

    async def process_txt_file(
        self,
        file_path: str,
        config: BatchJobConfig | None = None,
    ) -> BatchJobResult:
        """Verarbeitet eine TXT-Datei."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        job = self.processor.create_job(config)
        job.add_text(text, source_id=file_path)
        return await self.processor.process_job(job)

    async def process_csv_file(
        self,
        file_path: str,
        text_column: str = "text",
        id_column: str | None = "id",
        config: BatchJobConfig | None = None,
    ) -> BatchJobResult:
        """Verarbeitet eine CSV-Datei."""
        import csv

        job = self.processor.create_job(config)

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get(text_column, "")
                source_id = row.get(id_column) if id_column else None
                if text.strip():
                    job.add_text(text, source_id=source_id)

        return await self.processor.process_job(job)

    async def process_jsonl_file(
        self,
        file_path: str,
        text_field: str = "text",
        id_field: str | None = "id",
        config: BatchJobConfig | None = None,
    ) -> BatchJobResult:
        """Verarbeitet eine JSONL-Datei (ein JSON pro Zeile)."""
        import json

        job = self.processor.create_job(config)

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    text = (
                        obj.get(text_field, "") if isinstance(obj, dict) else str(obj)
                    )
                    source_id = (
                        obj.get(id_field)
                        if isinstance(obj, dict) and id_field
                        else None
                    )
                    if text.strip():
                        job.add_text(text, source_id=source_id)

        return await self.processor.process_job(job)
