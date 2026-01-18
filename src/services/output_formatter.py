"""
Veritas Output Formatter

Formatiert Analyse-Ergebnisse fuer verschiedene Ausgabeformate:
- Terminal (ASCII)
- HTML
- Markdown
- JSON
"""

from typing import Optional
from src.models.veritas_schema import (
    FullAnalysis,
    ClaimAnalysis,
    FactStatus,
    ContextStatus,
    NarrativeStatus,
    VerdictType,
    VERDICT_DESCRIPTIONS,
    HistoricalMyth,
)


# =============================================================================
# Status Icons und Farben
# =============================================================================

STATUS_ICONS = {
    # Fact Status
    FactStatus.CONFIRMED: ("✓", "green"),
    FactStatus.LIKELY: ("○", "green"),
    FactStatus.DISPUTED: ("◐", "yellow"),
    FactStatus.UNVERIFIED: ("?", "gray"),
    FactStatus.FALSE: ("✗", "red"),
    FactStatus.MYTH: ("⚠", "orange"),
    # Context Status
    ContextStatus.COMPLETE: ("✓", "green"),
    ContextStatus.SIMPLIFIED: ("○", "yellow"),
    ContextStatus.SELECTIVE: ("◐", "orange"),
    ContextStatus.DECONTEXTUALIZED: ("!", "orange"),
    ContextStatus.MISLEADING: ("✗", "red"),
    # Narrative Status
    NarrativeStatus.NEUTRAL: ("○", "green"),
    NarrativeStatus.PERSPECTIVAL: ("◐", "yellow"),
    NarrativeStatus.BIASED: ("!", "orange"),
    NarrativeStatus.PROPAGANDA: ("⚠", "red"),
    NarrativeStatus.REVISIONISM: ("✗", "red"),
}

VERDICT_ICONS = {
    "historically_accurate": ("✓", "green", "Historisch korrekt"),
    "simplified": ("○", "yellow", "Vereinfacht"),
    "selective": ("◐", "orange", "Selektiv"),
    "decontextualized": ("!", "orange", "Dekontextualisiert"),
    "reinterpreted": ("~", "orange", "Umgedeutet"),
    "propaganda": ("⚠", "red", "Propaganda"),
    "false": ("✗", "red", "Falsch"),
    "unverifiable": ("?", "gray", "Nicht verifizierbar"),
}


# =============================================================================
# ASCII Formatter (Terminal)
# =============================================================================


def format_analysis_ascii(analysis: FullAnalysis, width: int = 70) -> str:
    """Formatiert Analyse fuer Terminal-Ausgabe."""
    lines = []
    sep = "═" * width
    thin_sep = "─" * width

    # Header
    lines.append(sep)
    lines.append(center_text("VERITAS ANALYSE", width))
    lines.append(sep)
    lines.append("")

    # Input
    lines.append("EINGABE:")
    lines.append(wrap_text(analysis.input_text, width - 2, prefix="  "))
    lines.append("")

    # Verdict
    icon, color, label = VERDICT_ICONS.get(
        analysis.overall_verdict, ("?", "gray", "Unbekannt")
    )
    lines.append(thin_sep)
    lines.append(f"BEWERTUNG: {icon} {label.upper()}")
    lines.append(thin_sep)
    lines.append("")

    # Explanation
    if analysis.verdict_explanation:
        lines.append(wrap_text(analysis.verdict_explanation, width))
        lines.append("")

    # Claims
    if analysis.claims:
        lines.append(thin_sep)
        lines.append("ANALYSE DER BEHAUPTUNGEN:")
        lines.append(thin_sep)

        for i, claim in enumerate(analysis.claims, 1):
            icon, _ = STATUS_ICONS.get(claim.fact_status, ("?", "gray"))
            lines.append(
                f"\n{i}. {claim.original_claim[:60]}{'...' if len(claim.original_claim) > 60 else ''}"
            )
            lines.append(f"   Status: {icon} {claim.fact_status.value}")
            lines.append(f"   Konfidenz: {claim.confidence_score:.0%}")

            if claim.what_is_true:
                lines.append("   ✓ Was stimmt:")
                for item in claim.what_is_true[:3]:
                    lines.append(f"     - {item}")

            if claim.what_is_false:
                lines.append("   ✗ Was nicht stimmt:")
                for item in claim.what_is_false[:3]:
                    lines.append(f"     - {item}")

    # Context
    if analysis.context_analysis:
        ctx = analysis.context_analysis
        if ctx.important_omissions or ctx.missing_perspectives:
            lines.append("")
            lines.append(thin_sep)
            lines.append("FEHLENDER KONTEXT:")
            lines.append(thin_sep)

            if ctx.missing_timeframe:
                lines.append(f"  Zeitraum: {ctx.missing_timeframe}")

            for omission in ctx.important_omissions[:3]:
                lines.append(f"  - {omission}")

            for perspective in ctx.missing_perspectives[:3]:
                lines.append(f"  - Perspektive fehlt: {perspective}")

    # Summary for Users
    lines.append("")
    lines.append(thin_sep)
    lines.append("ZUSAMMENFASSUNG:")
    lines.append(thin_sep)
    lines.append(wrap_text(analysis.summary_for_users, width))

    # Recommendation
    if analysis.recommendation:
        lines.append("")
        lines.append(f"💡 {analysis.recommendation}")

    # Sources
    if analysis.all_sources:
        lines.append("")
        lines.append(thin_sep)
        lines.append("QUELLEN:")
        lines.append(thin_sep)
        for source in analysis.all_sources[:5]:
            author = f"{source.author}, " if source.author else ""
            year = f"({source.year})" if source.year else ""
            lines.append(f"  - {author}{source.title} {year}")

    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


def format_myth_ascii(myth: HistoricalMyth, width: int = 70) -> str:
    """Formatiert einen Mythos fuer Terminal-Ausgabe."""
    lines = []
    sep = "═" * width
    thin_sep = "─" * width

    icon, _ = STATUS_ICONS.get(myth.status, ("?", "gray"))

    lines.append(sep)
    lines.append(center_text(f"MYTHOS: {myth.id}", width))
    lines.append(sep)
    lines.append("")

    lines.append(f"BEHAUPTUNG: {icon}")
    lines.append(wrap_text(myth.claim, width))
    lines.append("")

    lines.append(thin_sep)
    lines.append("WAHRHEIT:")
    lines.append(thin_sep)
    lines.append(wrap_text(myth.truth, width))
    lines.append("")

    lines.append(thin_sep)
    lines.append("URSPRUNG:")
    lines.append(thin_sep)
    lines.append(f"  Quelle: {myth.origin.source}")
    if myth.origin.date:
        lines.append(f"  Zeit: {myth.origin.date}")
    lines.append(f"  Grund: {myth.origin.reason}")

    if myth.sources:
        lines.append("")
        lines.append(thin_sep)
        lines.append("QUELLEN:")
        lines.append(thin_sep)
        for source in myth.sources[:3]:
            lines.append(f"  - {source.title}")

    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


# =============================================================================
# Markdown Formatter
# =============================================================================


def format_analysis_markdown(analysis: FullAnalysis) -> str:
    """Formatiert Analyse als Markdown."""
    lines = []

    # Header
    lines.append("# Veritas Analyse")
    lines.append("")

    # Verdict Badge
    icon, color, label = VERDICT_ICONS.get(
        analysis.overall_verdict, ("?", "gray", "Unbekannt")
    )
    lines.append(f"**Bewertung:** {icon} **{label.upper()}**")
    lines.append("")

    # Input
    lines.append("## Eingabe")
    lines.append(f"> {analysis.input_text}")
    lines.append("")

    # Explanation
    if analysis.verdict_explanation:
        lines.append("## Erklaerung")
        lines.append(analysis.verdict_explanation)
        lines.append("")

    # Claims
    if analysis.claims:
        lines.append("## Analysierte Behauptungen")
        lines.append("")

        for i, claim in enumerate(analysis.claims, 1):
            icon, _ = STATUS_ICONS.get(claim.fact_status, ("?", "gray"))
            lines.append(f"### {i}. {claim.original_claim[:80]}")
            lines.append("")
            lines.append(f"- **Status:** {icon} {claim.fact_status.value}")
            lines.append(f"- **Konfidenz:** {claim.confidence_score:.0%}")

            if claim.what_is_true:
                lines.append("")
                lines.append("**Was stimmt:**")
                for item in claim.what_is_true:
                    lines.append(f"- ✓ {item}")

            if claim.what_is_false:
                lines.append("")
                lines.append("**Was nicht stimmt:**")
                for item in claim.what_is_false:
                    lines.append(f"- ✗ {item}")

            lines.append("")

    # Context
    if analysis.context_analysis:
        ctx = analysis.context_analysis
        if ctx.important_omissions or ctx.missing_perspectives:
            lines.append("## Fehlender Kontext")
            lines.append("")

            if ctx.missing_timeframe:
                lines.append(f"**Zeitraum:** {ctx.missing_timeframe}")
                lines.append("")

            if ctx.important_omissions:
                lines.append("**Wichtige Auslassungen:**")
                for item in ctx.important_omissions:
                    lines.append(f"- {item}")
                lines.append("")

            if ctx.missing_perspectives:
                lines.append("**Fehlende Perspektiven:**")
                for item in ctx.missing_perspectives:
                    lines.append(f"- {item}")
                lines.append("")

    # Summary
    lines.append("## Zusammenfassung")
    lines.append(analysis.summary_for_users)
    lines.append("")

    if analysis.recommendation:
        lines.append(f"> 💡 **Empfehlung:** {analysis.recommendation}")
        lines.append("")

    # Sources
    if analysis.all_sources:
        lines.append("## Quellen")
        lines.append("")
        for source in analysis.all_sources:
            author = f"{source.author}. " if source.author else ""
            year = f" ({source.year})" if source.year else ""
            url = f" [Link]({source.url})" if source.url else ""
            lines.append(f"- {author}*{source.title}*{year}{url}")

    return "\n".join(lines)


# =============================================================================
# HTML Formatter
# =============================================================================


def format_analysis_html(analysis: FullAnalysis) -> str:
    """Formatiert Analyse als HTML."""
    icon, color, label = VERDICT_ICONS.get(
        analysis.overall_verdict, ("?", "gray", "Unbekannt")
    )

    # CSS Farben
    color_map = {
        "green": "#22c55e",
        "yellow": "#eab308",
        "orange": "#f97316",
        "red": "#ef4444",
        "gray": "#6b7280",
    }
    verdict_color = color_map.get(color, "#6b7280")

    html = f"""
<div class="veritas-analysis">
    <div class="verdict-badge" style="background-color: {verdict_color}20; border-left: 4px solid {verdict_color}; padding: 1rem; margin-bottom: 1rem;">
        <span style="font-size: 1.5rem;">{icon}</span>
        <strong style="color: {verdict_color}; margin-left: 0.5rem;">{label.upper()}</strong>
    </div>
    
    <div class="input-section">
        <h4>Eingabe</h4>
        <blockquote>{analysis.input_text}</blockquote>
    </div>
    
    <div class="explanation-section">
        <h4>Erklaerung</h4>
        <p>{analysis.verdict_explanation}</p>
    </div>
"""

    # Claims
    if analysis.claims:
        html += '<div class="claims-section"><h4>Analysierte Behauptungen</h4>'
        for claim in analysis.claims:
            c_icon, c_color = STATUS_ICONS.get(claim.fact_status, ("?", "gray"))
            c_hex = color_map.get(c_color, "#6b7280")

            html += f"""
            <div class="claim-card" style="border-left: 3px solid {c_hex}; padding: 0.5rem 1rem; margin: 0.5rem 0; background: #f8f9fa;">
                <p><strong>{c_icon} {claim.original_claim}</strong></p>
                <p>Status: {claim.fact_status.value} | Konfidenz: {claim.confidence_score:.0%}</p>
            </div>
"""
        html += "</div>"

    # Summary
    html += f"""
    <div class="summary-section" style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h4>Zusammenfassung</h4>
        <p>{analysis.summary_for_users}</p>
        {f'<p><em>💡 {analysis.recommendation}</em></p>' if analysis.recommendation else ''}
    </div>
</div>
"""

    return html


# =============================================================================
# Helper Functions
# =============================================================================


def center_text(text: str, width: int) -> str:
    """Zentriert Text."""
    padding = (width - len(text)) // 2
    return " " * padding + text


def wrap_text(text: str, width: int, prefix: str = "") -> str:
    """Bricht Text um."""
    words = text.split()
    lines = []
    current_line = prefix

    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line += (" " if current_line != prefix else "") + word
        else:
            if current_line != prefix:
                lines.append(current_line)
            current_line = prefix + word

    if current_line != prefix:
        lines.append(current_line)

    return "\n".join(lines)
