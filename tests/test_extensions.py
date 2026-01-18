"""
Tests für die erweiterten Module:
- Batch Processing
- Validation
- ML Confidence
- Data Sources
"""

import pytest
from datetime import date
from uuid import uuid4

# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchProcessing:
    """Tests für das Batch-Processing Modul."""

    def test_batch_job_config_defaults(self):
        """Test default configuration values."""
        from src.processing.batch import BatchJobConfig

        config = BatchJobConfig()
        assert config.max_concurrent == 5
        assert config.max_retries == 3
        assert config.skip_duplicates is True
        assert config.store_results is True

    def test_batch_job_config_custom(self):
        """Test custom configuration."""
        from src.processing.batch import BatchJobConfig

        config = BatchJobConfig(
            max_concurrent=10,
            max_retries=5,
            timeout_seconds=120.0,
            as_facts=True,
        )
        assert config.max_concurrent == 10
        assert config.max_retries == 5
        assert config.as_facts is True

    def test_batch_item_hash(self):
        """Test text hashing for duplicate detection."""
        from src.processing.batch import BatchItem

        item1 = BatchItem(text="Hello World")
        item2 = BatchItem(text="Hello World")
        item3 = BatchItem(text="Different Text")

        assert item1.text_hash == item2.text_hash
        assert item1.text_hash != item3.text_hash

    def test_batch_job_add_texts(self):
        """Test adding texts to a batch job."""
        from src.processing.batch import BatchJob, BatchJobConfig

        job = BatchJob(config=BatchJobConfig())
        items = job.add_texts(["Text 1", "Text 2", "Text 3"])

        assert len(items) == 3
        assert len(job.items) == 3
        assert all(item.text for item in items)

    def test_batch_job_progress(self):
        """Test progress calculation."""
        from src.processing.batch import BatchJob, BatchItem, ItemStatus

        job = BatchJob()
        job.items = [
            BatchItem(text="t1", status=ItemStatus.SUCCESS),
            BatchItem(text="t2", status=ItemStatus.SUCCESS),
            BatchItem(text="t3", status=ItemStatus.FAILED),
            BatchItem(text="t4", status=ItemStatus.QUEUED),
        ]

        progress = job.progress
        assert progress.total_items == 4
        assert progress.processed_items == 3
        assert progress.successful_items == 2
        assert progress.failed_items == 1
        assert progress.progress_percent == 75.0


# =============================================================================
# Validation Tests
# =============================================================================


class TestEntityResolver:
    """Tests für Entity Resolution."""

    def test_normalize_text(self):
        """Test text normalization."""
        from src.validation.validator import EntityResolver

        resolver = EntityResolver()

        assert resolver._normalize("  Napoleon Bonaparte  ") == "napoleon bonaparte"
        assert resolver._normalize("LUDWIG XVI.") == "ludwig xvi"

    def test_levenshtein_ratio(self):
        """Test Levenshtein similarity calculation."""
        from src.validation.validator import EntityResolver

        resolver = EntityResolver()

        # Identical strings
        assert resolver._levenshtein_ratio("test", "test") == 1.0

        # Similar strings
        ratio = resolver._levenshtein_ratio("napoleon", "napoleón")
        assert 0.8 < ratio < 1.0

        # Different strings
        ratio = resolver._levenshtein_ratio("caesar", "napoleon")
        assert ratio < 0.5

    def test_soundex(self):
        """Test Soundex phonetic encoding."""
        from src.validation.validator import EntityResolver

        resolver = EntityResolver()

        # Similar sounding names should have same Soundex
        assert resolver._soundex("Robert") == resolver._soundex("Rupert")
        assert resolver._soundex("Smith") == resolver._soundex("Smythe")


class TestChronologyValidator:
    """Tests für chronologische Validierung."""

    def test_lifespan_validation_valid(self):
        """Test valid lifespan."""
        from src.validation.validator import ChronologyValidator
        from src.models.schema import PersonNode

        validator = ChronologyValidator()

        person = PersonNode(
            name="Test Person",
            birth_date=date(1900, 1, 1),
            death_date=date(1980, 12, 31),
        )

        # Event within lifespan
        issues = validator._check_person_lifespan(
            person,
            [(date(1950, 6, 15), "Some Event")],
        )

        assert len(issues) == 0

    def test_lifespan_validation_birth_after_death(self):
        """Test invalid lifespan (birth after death)."""
        from src.validation.validator import ChronologyValidator, IssueSeverity
        from src.models.schema import PersonNode

        validator = ChronologyValidator()

        person = PersonNode(
            name="Test Person",
            birth_date=date(1980, 1, 1),
            death_date=date(1900, 12, 31),
        )

        issues = validator._check_person_lifespan(person, [])

        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.CRITICAL

    def test_lifespan_validation_event_before_birth(self):
        """Test event before person's birth."""
        from src.validation.validator import ChronologyValidator, IssueSeverity
        from src.models.schema import PersonNode

        validator = ChronologyValidator()

        person = PersonNode(
            name="Test Person",
            birth_date=date(1900, 1, 1),
            death_date=date(1980, 12, 31),
        )

        issues = validator._check_person_lifespan(
            person,
            [(date(1850, 6, 15), "Event Before Birth")],
        )

        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.CRITICAL
        assert "could not have participated" in issues[0].message


# =============================================================================
# ML Confidence Tests
# =============================================================================


class TestFeatureExtractor:
    """Tests für Feature Extraction."""

    def test_extract_structural_features(self):
        """Test structural feature extraction."""
        from src.ml.confidence import FeatureExtractor, ClaimFeatures
        from src.models.schema import (
            KnowledgeGraphExtraction,
            PersonNode,
            EventNode,
            Relationship,
            RelationType,
        )

        extractor = FeatureExtractor()

        extraction = KnowledgeGraphExtraction(
            nodes=[
                PersonNode(name="Napoleon", birth_date=date(1769, 8, 15)),
                PersonNode(name="Josephine"),
                EventNode(name="Battle of Waterloo", start_date=date(1815, 6, 18)),
            ],
            relationships=[
                Relationship(
                    source_name="Napoleon",
                    target_name="Battle of Waterloo",
                    relation_type=RelationType.PARTICIPATED_IN,
                ),
            ],
        )

        features = extractor.extract_from_extraction(extraction)

        assert features.num_nodes == 3
        assert features.num_persons == 2
        assert features.num_events == 1
        assert features.num_relationships == 1

    def test_completeness_score(self):
        """Test completeness score calculation."""
        from src.ml.confidence import FeatureExtractor, ClaimFeatures

        extractor = FeatureExtractor()

        # High completeness
        features = ClaimFeatures(
            num_nodes=10,
            nodes_with_dates=0.8,
            relationships_per_node=1.5,
            avg_description_length=150,
        )
        score = extractor._calc_completeness(features)
        assert score > 0.7

        # Low completeness
        features = ClaimFeatures(
            num_nodes=1,
            nodes_with_dates=0.0,
            relationships_per_node=0.0,
            avg_description_length=0,
        )
        score = extractor._calc_completeness(features)
        assert score < 0.3


class TestConfidenceScorer:
    """Tests für Confidence Scoring."""

    def test_rule_based_scorer(self):
        """Test rule-based scoring."""
        from src.ml.confidence import RuleBasedScorer, ClaimFeatures

        scorer = RuleBasedScorer()

        # Good features
        features = ClaimFeatures(
            entity_match_rate=0.9,
            completeness_score=0.8,
            specificity_score=0.7,
            consistency_score=0.8,
            num_critical_issues=0,
        )
        score, contributions = scorer.score(features)
        assert score > 0.6

        # Bad features (critical issues)
        features = ClaimFeatures(
            entity_match_rate=0.5,
            num_critical_issues=3,
            dates_in_future=True,
        )
        score, contributions = scorer.score(features)
        assert score < 0.3

    def test_ensemble_scorer_untrained(self):
        """Test ensemble scorer without training."""
        from src.ml.confidence import EnsembleConfidenceScorer, ClaimFeatures

        scorer = EnsembleConfidenceScorer()

        features = ClaimFeatures(
            num_nodes=5,
            entity_match_rate=0.8,
            completeness_score=0.7,
        )

        result = scorer.score(features, explain=True)

        assert "confidence" in result
        assert "components" in result
        assert "explanation" in result
        assert not result["is_trained"]


# =============================================================================
# Data Sources Tests
# =============================================================================


class TestDataSources:
    """Tests für externe Datenquellen."""

    def test_wikidata_source_init(self):
        """Test Wikidata source initialization."""
        from src.datasources.external import WikidataSource

        source = WikidataSource()
        assert source.config.source_type.value == "wikidata"

    def test_dbpedia_source_init(self):
        """Test DBpedia source initialization."""
        from src.datasources.external import DBpediaSource

        source = DBpediaSource()
        assert source.config.source_type.value == "dbpedia"

    def test_data_source_manager_register(self):
        """Test registering data sources."""
        from src.datasources.external import (
            DataSourceManager,
            WikidataSource,
            DataSourceType,
        )

        manager = DataSourceManager()
        manager.register_source(WikidataSource())

        assert DataSourceType.WIKIDATA in manager.sources

    def test_data_source_manager_defaults(self):
        """Test registering default sources."""
        from src.datasources.external import DataSourceManager, DataSourceType

        manager = DataSourceManager()
        manager.register_defaults()

        assert DataSourceType.WIKIDATA in manager.sources
        assert DataSourceType.DBPEDIA in manager.sources


# =============================================================================
# Integration Tests (require running services)
# =============================================================================


@pytest.mark.skip(reason="Requires external API access")
class TestWikidataIntegration:
    """Integration tests für Wikidata."""

    @pytest.mark.asyncio
    async def test_search_person(self):
        """Test searching for a person."""
        from src.datasources.external import WikidataSource

        source = WikidataSource()
        results = await source.search_person("Napoleon")

        assert len(results) > 0
        await source.close()

    @pytest.mark.asyncio
    async def test_import_to_graph(self):
        """Test importing to graph."""
        from src.datasources.external import WikidataSource
        from src.models.schema import NodeType

        source = WikidataSource()
        result = await source.import_to_graph("Napoleon", NodeType.PERSON)

        assert result.success
        assert result.nodes_imported > 0
        await source.close()
