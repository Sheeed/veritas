"""Data sources integration."""

from .external import (
    DataSourceManager,
    WikidataSource,
    DBpediaSource,
    DataSourceType,
    ImportResult,
)
from .authority import (
    AuthoritySourceManager,
    GNDSource,
    VIAFSource,
    LOCSource,
    GettySource,
    AuthoritySourceType,
    AuthorityRecord,
    AuthorityImportResult,
)

__all__ = [
    # External (legacy, weniger zuverlässig)
    "DataSourceManager",
    "WikidataSource",
    "DBpediaSource",
    "DataSourceType",
    "ImportResult",
    # Authority (empfohlen, höchste Qualität)
    "AuthoritySourceManager",
    "GNDSource",
    "VIAFSource",
    "LOCSource",
    "GettySource",
    "AuthoritySourceType",
    "AuthorityRecord",
    "AuthorityImportResult",
]
