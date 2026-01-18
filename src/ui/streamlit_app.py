"""
History Guardian - Streamlit UI
Design inspiriert von Claude/Anthropic: Clean, minimalistisch, professionell
"""

import asyncio
import json
from datetime import datetime
from typing import Any

import streamlit as st
import httpx

# =============================================================================
# Konfiguration
# =============================================================================

st.set_page_config(
    page_title="History Guardian",
    page_icon="HG",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = "http://localhost:8000"

# =============================================================================
# Claude-inspiriertes CSS Design
# =============================================================================

st.markdown(
    """
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Root Variables */
    :root {
        --bg-primary: #1a1a1a;
        --bg-secondary: #2d2d2d;
        --bg-tertiary: #3d3d3d;
        --text-primary: #fafafa;
        --text-secondary: #a1a1a1;
        --text-muted: #737373;
        --accent: #d97706;
        --accent-hover: #f59e0b;
        --success: #22c55e;
        --warning: #eab308;
        --error: #ef4444;
        --border: #404040;
    }
    
    /* Global Styles */
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main .block-container {
        max-width: 900px;
        padding: 2rem 1rem;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    p, span, label {
        color: var(--text-secondary);
    }
    
    /* Header */
    .hg-header {
        text-align: center;
        padding: 3rem 0 2rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    
    .hg-logo {
        font-size: 1rem;
        font-weight: 600;
        color: var(--accent);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    .hg-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        line-height: 1.2;
    }
    
    .hg-subtitle {
        font-size: 1rem;
        color: var(--text-muted);
        margin-top: 0.5rem;
    }
    
    /* Input Area */
    .stTextArea textarea {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(217, 119, 6, 0.2) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background-color: var(--accent-hover) !important;
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary Button */
    .secondary-btn > button {
        background-color: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--text-secondary) !important;
    }
    
    .secondary-btn > button:hover {
        background-color: var(--bg-tertiary) !important;
        border-color: var(--text-muted) !important;
    }
    
    /* Result Cards */
    .result-card {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .result-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .result-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }
    
    .result-icon.success { background-color: rgba(34, 197, 94, 0.15); }
    .result-icon.warning { background-color: rgba(234, 179, 8, 0.15); }
    .result-icon.error { background-color: rgba(239, 68, 68, 0.15); }
    .result-icon.info { background-color: rgba(217, 119, 6, 0.15); }
    
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .result-subtitle {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin: 0;
    }
    
    /* Confidence Display */
    .confidence-display {
        text-align: center;
        padding: 1.5rem;
    }
    
    .confidence-value {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1;
    }
    
    .confidence-value.high { color: var(--success); }
    .confidence-value.medium { color: var(--warning); }
    .confidence-value.low { color: var(--error); }
    
    .confidence-label {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin-top: 0.5rem;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background-color: var(--accent) !important;
    }
    
    .stProgress {
        background-color: var(--bg-tertiary);
    }
    
    /* Entity List */
    .entity-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background-color: var(--bg-tertiary);
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .entity-type {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .entity-name {
        font-size: 1rem;
        color: var(--text-primary);
        font-weight: 500;
    }
    
    .entity-meta {
        font-size: 0.875rem;
        color: var(--text-muted);
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.375rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-verified {
        background-color: rgba(34, 197, 94, 0.15);
        color: var(--success);
    }
    
    .status-suspicious {
        background-color: rgba(234, 179, 8, 0.15);
        color: var(--warning);
    }
    
    .status-contradicted {
        background-color: rgba(239, 68, 68, 0.15);
        color: var(--error);
    }
    
    /* Issue Cards */
    .issue-card {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border-left: 3px solid;
    }
    
    .issue-critical {
        background-color: rgba(239, 68, 68, 0.1);
        border-color: var(--error);
    }
    
    .issue-warning {
        background-color: rgba(234, 179, 8, 0.1);
        border-color: var(--warning);
    }
    
    .issue-info {
        background-color: rgba(217, 119, 6, 0.1);
        border-color: var(--accent);
    }
    
    .issue-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .issue-message {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: var(--bg-secondary);
        border-radius: 8px;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: var(--text-muted);
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Divider */
    .hg-divider {
        height: 1px;
        background-color: var(--border);
        margin: 2rem 0;
    }
    
    /* Navigation */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        padding: 1rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    
    .nav-item {
        padding: 0.5rem 1rem;
        color: var(--text-muted);
        text-decoration: none;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .nav-item:hover {
        color: var(--text-primary);
        background-color: var(--bg-secondary);
    }
    
    .nav-item.active {
        color: var(--accent);
        background-color: rgba(217, 119, 6, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--bg-secondary) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--accent) transparent transparent transparent !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
    }
    
    /* JSON */
    .stJson {
        background-color: var(--bg-secondary) !important;
        border-radius: 8px;
    }
    
    /* Alerts */
    .stAlert {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    
    /* Radio Buttons */
    .stRadio > label {
        color: var(--text-secondary) !important;
    }
    
    .stRadio [data-baseweb="radio"] {
        background-color: var(--bg-tertiary) !important;
    }
    
    /* Checkbox */
    .stCheckbox span {
        color: var(--text-secondary) !important;
    }
    
    /* Select */
    .stSelectbox [data-baseweb="select"] {
        background-color: var(--bg-secondary) !important;
    }
    
    /* Footer */
    .hg-footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid var(--border);
        color: var(--text-muted);
        font-size: 0.8rem;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# API Client
# =============================================================================


class HistoryGuardianClient:
    """Async Client fuer die History Guardian API."""

    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url

    async def health_check(self) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            return response.json()

    async def get_stats(self) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/stats")
            return response.json()

    async def extract(self, text: str) -> dict:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/extract",
                json={"text": text},
            )
            return response.json()

    async def validate(self, extraction: dict) -> dict:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/validate",
                json={"extraction": extraction},
            )
            return response.json()

    async def ingest(self, text: str, as_fact: bool = False) -> dict:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/ingest",
                json={"text": text, "as_fact": as_fact},
            )
            return response.json()

    async def score_confidence(self, extraction: dict, source_text: str = None) -> dict:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/score/confidence",
                json={"extraction": extraction, "source_text": source_text},
            )
            return response.json()


def run_async(coro):
    """Fuehrt eine Coroutine synchron aus."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


client = HistoryGuardianClient()


# =============================================================================
# UI Components
# =============================================================================


def render_header():
    """Rendert den Header."""
    st.markdown(
        """
    <div class="hg-header">
        <div class="hg-logo">History Guardian</div>
        <h1 class="hg-title">Historische Fakten verifizieren</h1>
        <p class="hg-subtitle">GraphRAG-basierte Analyse mit autoritativen Quellen</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_navigation(current_page: str) -> str:
    """Rendert die Navigation."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(
            "Analyse",
            use_container_width=True,
            type="primary" if current_page == "analysis" else "secondary",
        ):
            return "analysis"
    with col2:
        if st.button(
            "Datenbank",
            use_container_width=True,
            type="primary" if current_page == "database" else "secondary",
        ):
            return "database"
    with col3:
        if st.button(
            "Quellen",
            use_container_width=True,
            type="primary" if current_page == "sources" else "secondary",
        ):
            return "sources"
    with col4:
        if st.button(
            "Status",
            use_container_width=True,
            type="primary" if current_page == "status" else "secondary",
        ):
            return "status"

    return current_page


def render_confidence(confidence: float):
    """Rendert die Konfidenz-Anzeige."""
    percentage = int(confidence * 100)

    if confidence >= 0.8:
        level = "high"
        label = "Hohe Konfidenz"
    elif confidence >= 0.5:
        level = "medium"
        label = "Mittlere Konfidenz"
    else:
        level = "low"
        label = "Niedrige Konfidenz"

    st.markdown(
        f"""
    <div class="confidence-display">
        <div class="confidence-value {level}">{percentage}%</div>
        <div class="confidence-label">{label}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.progress(confidence)


def render_status_badge(status: str):
    """Rendert einen Status-Badge."""
    status_map = {
        "verified": ("Verifiziert", "status-verified"),
        "contradicted": ("Widerlegt", "status-contradicted"),
        "suspicious": ("Verdaechtig", "status-suspicious"),
        "partially_verified": ("Teilweise verifiziert", "status-suspicious"),
        "unverifiable": ("Nicht pruefbar", "status-suspicious"),
    }

    label, css_class = status_map.get(status, ("Unbekannt", "status-suspicious"))
    st.markdown(
        f'<span class="status-badge {css_class}">{label}</span>', unsafe_allow_html=True
    )


def render_entity(node: dict):
    """Rendert eine Entitaet."""
    node_type = node.get("node_type", "Unknown")
    name = node.get("name", "Unbekannt")
    confidence = node.get("confidence", 0)

    extra_info = []
    if node.get("birth_date"):
        extra_info.append(f"geb. {node['birth_date']}")
    if node.get("death_date"):
        extra_info.append(f"gest. {node['death_date']}")
    if node.get("start_date"):
        extra_info.append(f"{node['start_date']}")

    meta = " | ".join(extra_info) if extra_info else f"Konfidenz: {confidence:.0%}"

    st.markdown(
        f"""
    <div class="entity-item">
        <div>
            <div class="entity-type">{node_type}</div>
            <div class="entity-name">{name}</div>
            <div class="entity-meta">{meta}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_issue(issue: dict):
    """Rendert ein Validierungsproblem."""
    severity = issue.get("severity", "info")
    issue_type = issue.get("issue_type", "")
    message = issue.get("message", "")

    severity_class = {
        "critical": "issue-critical",
        "high": "issue-critical",
        "medium": "issue-warning",
        "low": "issue-info",
        "info": "issue-info",
    }.get(severity, "issue-info")

    st.markdown(
        f"""
    <div class="issue-card {severity_class}">
        <div class="issue-title">{severity.upper()}: {issue_type}</div>
        <div class="issue-message">{message}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_divider():
    """Rendert einen Divider."""
    st.markdown('<div class="hg-divider"></div>', unsafe_allow_html=True)


def render_footer():
    """Rendert den Footer."""
    st.markdown(
        """
    <div class="hg-footer">
        History Guardian v0.3.0 | Powered by Groq LLM | Autoritative Quellen: GND, VIAF, LOC, Getty
    </div>
    """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Pages
# =============================================================================


def page_analysis():
    """Haupt-Analyse-Seite."""

    # Input Section
    st.markdown("### Behauptung eingeben")

    claim_text = st.text_area(
        "claim_input",
        height=150,
        placeholder="Geben Sie eine historische Behauptung ein, z.B.:\n\nNapoleon Bonaparte wurde am 15. August 1769 auf Korsika geboren und starb am 5. Mai 1821 auf St. Helena.",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_btn = st.button("Analysieren", type="primary", use_container_width=True)
    with col2:
        with st.popover("Optionen"):
            save_to_db = st.checkbox("In Datenbank speichern", value=False)
            show_details = st.checkbox("Details anzeigen", value=True)

    # Analysis
    if analyze_btn and claim_text:
        render_divider()

        # Progress indicators
        progress_placeholder = st.empty()
        result_placeholder = st.container()

        with progress_placeholder.container():
            st.markdown("**Analyse laeuft...**")
            progress_bar = st.progress(0)

        try:
            # Step 1: Extract
            progress_bar.progress(20)
            extract_result = run_async(client.extract(claim_text))

            if not extract_result.get("success"):
                st.error("Extraktion fehlgeschlagen. Bitte versuchen Sie es erneut.")
                return

            extraction = extract_result.get("extraction", {})
            nodes = extraction.get("nodes", [])
            relationships = extraction.get("relationships", [])

            # Step 2: Validate
            progress_bar.progress(50)
            validation_result = run_async(client.validate(extraction))

            # Step 3: ML Confidence
            progress_bar.progress(80)
            confidence_result = run_async(
                client.score_confidence(extraction, claim_text)
            )

            progress_bar.progress(100)
            progress_placeholder.empty()

            # Results
            with result_placeholder:
                # Summary Card
                status = validation_result.get("overall_status", "unverifiable")
                confidence = confidence_result.get("confidence", 0.5)

                st.markdown(
                    f"""
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-icon {'success' if status == 'verified' else 'warning' if status == 'suspicious' else 'error' if status == 'contradicted' else 'info'}">
                            {'✓' if status == 'verified' else '!' if status in ['suspicious', 'contradicted'] else '?'}
                        </div>
                        <div>
                            <p class="result-title">Analyseergebnis</p>
                            <p class="result-subtitle">{len(nodes)} Entitaeten, {len(relationships)} Beziehungen extrahiert</p>
                        </div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Main Results
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Bewertung")
                    render_status_badge(status)
                    st.markdown(
                        f"**{validation_result.get('summary', 'Keine Zusammenfassung verfuegbar')}**"
                    )

                    if validation_result.get("recommendation"):
                        st.info(validation_result["recommendation"])

                with col2:
                    st.markdown("#### Konfidenz")
                    render_confidence(confidence)

                # Detailed Results
                if show_details:
                    render_divider()

                    tab1, tab2, tab3 = st.tabs(
                        ["Entitaeten", "Beziehungen", "Probleme"]
                    )

                    with tab1:
                        if nodes:
                            for node in nodes:
                                render_entity(node)
                        else:
                            st.markdown("*Keine Entitaeten extrahiert*")

                    with tab2:
                        if relationships:
                            for rel in relationships:
                                source = rel.get("source_name", "?")
                                target = rel.get("target_name", "?")
                                rel_type = rel.get("relation_type", "?")
                                st.markdown(
                                    f"**{source}** → *{rel_type}* → **{target}**"
                                )
                        else:
                            st.markdown("*Keine Beziehungen extrahiert*")

                    with tab3:
                        issues = validation_result.get("all_issues", [])
                        if issues:
                            for issue in issues:
                                render_issue(issue)
                        else:
                            st.success("Keine Probleme gefunden")

                # Save to DB
                if save_to_db:
                    ingest_result = run_async(client.ingest(claim_text, as_fact=False))
                    if ingest_result.get("success"):
                        st.success("Als Claim in Datenbank gespeichert")

        except httpx.ConnectError:
            progress_placeholder.empty()
            st.error("Verbindung zur API fehlgeschlagen. Ist der Server gestartet?")
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"Fehler bei der Analyse: {str(e)}")


def page_database():
    """Datenbank-Verwaltungs-Seite."""
    st.markdown("### Datenbank")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Fakt hinzufuegen")
        st.markdown("*Verifizierte historische Fakten als Ground Truth speichern*")

        fact_text = st.text_area(
            "fact_input",
            height=120,
            placeholder="Geben Sie einen verifizierten historischen Fakt ein...",
            label_visibility="collapsed",
        )

        if st.button("Als Fakt speichern", use_container_width=True):
            if fact_text:
                with st.spinner("Speichere..."):
                    result = run_async(client.ingest(fact_text, as_fact=True))
                    if result.get("success"):
                        st.success(
                            f"Gespeichert: {result.get('nodes_added', 0)} Knoten"
                        )
                    else:
                        st.error("Speichern fehlgeschlagen")

    with col2:
        st.markdown("#### Statistiken")

        try:
            stats = run_async(client.get_stats())
            if stats.get("success"):
                statistics = stats.get("statistics", {})

                st.metric("Fakten", statistics.get("Fact", 0))
                st.metric("Claims", statistics.get("Claim", 0))
                st.metric("Personen", statistics.get("Person", 0))
                st.metric("Ereignisse", statistics.get("Event", 0))
        except Exception:
            st.warning("Statistiken nicht verfuegbar")


def page_sources():
    """Quellen-Uebersichts-Seite."""
    st.markdown("### Autoritative Quellen")
    st.markdown("*Alle Verifikationen basieren auf bibliothekarischen Normdateien*")

    render_divider()

    sources = [
        {
            "name": "GND",
            "full": "Gemeinsame Normdatei",
            "org": "Deutsche Nationalbibliothek",
            "quality": "Hoechste",
            "coverage": "Deutschsprachiger Raum, universal",
            "url": "https://gnd.network",
        },
        {
            "name": "VIAF",
            "full": "Virtual International Authority File",
            "org": "OCLC",
            "quality": "Hoch",
            "coverage": "International, aggregiert",
            "url": "https://viaf.org",
        },
        {
            "name": "LOC",
            "full": "Library of Congress Authorities",
            "org": "Library of Congress",
            "quality": "Hoechste",
            "coverage": "US-Standard, international",
            "url": "https://id.loc.gov",
        },
        {
            "name": "Getty TGN",
            "full": "Thesaurus of Geographic Names",
            "org": "Getty Research Institute",
            "quality": "Hoechste",
            "coverage": "Geografische Namen weltweit",
            "url": "https://www.getty.edu/research/tools/vocabularies/tgn/",
        },
    ]

    for source in sources:
        st.markdown(
            f"""
        <div class="result-card">
            <div class="result-header">
                <div class="result-icon info">{source['name'][:2]}</div>
                <div>
                    <p class="result-title">{source['name']}</p>
                    <p class="result-subtitle">{source['full']}</p>
                </div>
            </div>
            <p style="color: var(--text-secondary); margin: 0.5rem 0;">
                <strong>Organisation:</strong> {source['org']}<br>
                <strong>Qualitaet:</strong> {source['quality']}<br>
                <strong>Abdeckung:</strong> {source['coverage']}
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    render_divider()

    st.markdown("#### Warum keine Wikipedia?")
    st.markdown(
        """
    Wikipedia ist eine wertvolle Ressource, aber fuer Faktenverifikation ungeeignet:
    
    - **Crowdsourced**: Jeder kann Inhalte aendern
    - **Keine Autoritaet**: Keine bibliothekarische Qualitaetskontrolle
    - **Instabil**: Inhalte aendern sich laufend
    
    Stattdessen verwenden wir ausschliesslich **kuratierte Normdateien**, 
    die von professionellen Bibliothekaren gepflegt werden.
    """
    )


def page_status():
    """System-Status-Seite."""
    st.markdown("### Systemstatus")

    render_divider()

    # API Health
    st.markdown("#### API")
    try:
        health = run_async(client.health_check())

        col1, col2 = st.columns(2)
        with col1:
            if health.get("neo4j_connected"):
                st.success("Neo4j verbunden")
            else:
                st.error("Neo4j nicht erreichbar")
        with col2:
            st.info(f"Version: {health.get('version', 'Unbekannt')}")

    except httpx.ConnectError:
        st.error("API Server nicht erreichbar")
        st.markdown("Starte den Server mit:")
        st.code("uvicorn src.api.main:app --reload --port 8000", language="bash")
        return
    except Exception as e:
        st.error(f"Fehler: {e}")
        return

    render_divider()

    # Stats
    st.markdown("#### Datenbank-Statistiken")
    try:
        stats = run_async(client.get_stats())
        if stats.get("success"):
            statistics = stats.get("statistics", {})

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Fakten", statistics.get("Fact", 0))
            with col2:
                st.metric("Claims", statistics.get("Claim", 0))
            with col3:
                st.metric("Personen", statistics.get("Person", 0))
            with col4:
                st.metric("Ereignisse", statistics.get("Event", 0))
    except Exception:
        st.warning("Statistiken nicht verfuegbar")

    render_divider()

    # Configuration
    st.markdown("#### Konfiguration")
    st.markdown(
        """
    | Einstellung | Wert |
    |-------------|------|
    | LLM Provider | Groq |
    | Modell | llama-3.3-70b-versatile |
    | Neo4j URI | bolt://localhost:7687 |
    | API Port | 8000 |
    | UI Port | 8501 |
    """
    )


# =============================================================================
# Main App
# =============================================================================


def main():
    """Hauptfunktion."""

    # Session State
    if "page" not in st.session_state:
        st.session_state.page = "analysis"

    # Header
    render_header()

    # Navigation
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button(
            "Analyse",
            use_container_width=True,
            type="primary" if st.session_state.page == "analysis" else "secondary",
        ):
            st.session_state.page = "analysis"
            st.rerun()
    with col2:
        if st.button(
            "Datenbank",
            use_container_width=True,
            type="primary" if st.session_state.page == "database" else "secondary",
        ):
            st.session_state.page = "database"
            st.rerun()
    with col3:
        if st.button(
            "Quellen",
            use_container_width=True,
            type="primary" if st.session_state.page == "sources" else "secondary",
        ):
            st.session_state.page = "sources"
            st.rerun()
    with col4:
        if st.button(
            "Status",
            use_container_width=True,
            type="primary" if st.session_state.page == "status" else "secondary",
        ):
            st.session_state.page = "status"
            st.rerun()

    render_divider()

    # Page Content
    if st.session_state.page == "analysis":
        page_analysis()
    elif st.session_state.page == "database":
        page_database()
    elif st.session_state.page == "sources":
        page_sources()
    elif st.session_state.page == "status":
        page_status()

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
