"""Veritas Services."""

from src.services.veritas_analyzer import VeritasAnalyzer, get_analyzer
from src.services.output_formatter import (
    format_analysis_ascii,
    format_analysis_markdown,
    format_analysis_html,
    format_myth_ascii,
)
from src.services.veritas_prompts import (
    HISTORIAN_SYSTEM_PROMPT,
    FACT_CHECKER_SYSTEM_PROMPT,
    build_fact_check_prompt,
    build_context_analysis_prompt,
    build_narrative_analysis_prompt,
)

__all__ = [
    "VeritasAnalyzer",
    "get_analyzer",
    "format_analysis_ascii",
    "format_analysis_markdown", 
    "format_analysis_html",
    "format_myth_ascii",
    "HISTORIAN_SYSTEM_PROMPT",
    "FACT_CHECKER_SYSTEM_PROMPT",
    "build_fact_check_prompt",
    "build_context_analysis_prompt",
    "build_narrative_analysis_prompt",
]
