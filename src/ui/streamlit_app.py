"""
Streamlit Web Interface für The History Guardian.

Professionelles Dashboard für:
- Faktenprüfung mit Konfidenz-Anzeige
- Batch-Verarbeitung mit Progress
- Datenquellen-Management
- Analyse-Reports
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
    page_title="The History Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #64748b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
    }
    .status-verified {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .status-contradicted {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .status-suspicious {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .entity-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.5rem;
    }
    .issue-critical {
        background-color: #fee2e2;
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }
    .issue-high {
        background-color: #ffedd5;
        border-left: 4px solid #ea580c;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background: #dbeafe;
        color: #1e40af;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# API Client
# =============================================================================


class HistoryGuardianClient:
    """Async Client für die History Guardian API."""
    
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
        async with httpx.AsyncClient(timeout=60.0) as client:
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
    
    async def import_from_authority(self, query: str, entity_type: str, sources: list) -> dict:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/import/authority",
                json={"query": query, "entity_type": entity_type, "sources": sources},
            )
            return response.json()


def run_async(coro):
    """Führt eine Coroutine synchron aus."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


client = HistoryGuardianClient()


# =============================================================================
# UI Components
# =============================================================================


def render_confidence_gauge(confidence: float):
    """Rendert eine Konfidenz-Anzeige."""
    percentage = int(confidence * 100)
    
    if confidence >= 0.8:
        color = "#22c55e"
        label = "Hoch"
    elif confidence >= 0.5:
        color = "#eab308"
        label = "Mittel"
    else:
        color = "#ef4444"
        label = "Niedrig"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 3rem; font-weight: bold; color: {color};">{percentage}%</div>
        <div style="font-size: 1rem; color: #64748b;">Konfidenz: {label}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(confidence)


def render_status_badge(status: str):
    """Rendert einen Status-Badge."""
    status_classes = {
        "verified": ("✓ Verifiziert", "status-verified"),
        "contradicted": ("✗ Widerlegt", "status-contradicted"),
        "suspicious": ("⚠ Verdächtig", "status-suspicious"),
        "partially_verified": ("◐ Teilweise verifiziert", "status-suspicious"),
        "unverifiable": ("? Nicht prüfbar", "status-suspicious"),
    }
    
    label, css_class = status_classes.get(status, ("Unbekannt", ""))
    st.markdown(f'<span class="{css_class}">{label}</span>', unsafe_allow_html=True)


def render_entity_card(node: dict):
    """Rendert eine Entity-Karte."""
    node_type = node.get("node_type", "Unknown")
    name = node.get("name", "Unbekannt")
    confidence = node.get("confidence", 0)
    
    type_icons = {
        "Person": "👤",
        "Event": "📅",
        "Location": "📍",
        "Date": "🗓️",
        "Organization": "🏛️",
    }
    
    icon = type_icons.get(node_type, "📌")
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{icon} {name}**")
            st.caption(f"{node_type} • Konfidenz: {confidence:.0%}")
        with col2:
            if node.get("birth_date"):
                st.caption(f"* {node['birth_date']}")
            if node.get("start_date"):
                st.caption(f"📅 {node['start_date']}")


def render_issue(issue: dict):
    """Rendert ein Validierungsproblem."""
    severity = issue.get("severity", "info")
    issue_type = issue.get("issue_type", "")
    message = issue.get("message", "")
    suggestion = issue.get("suggestion", "")
    
    severity_colors = {
        "critical": ("#fee2e2", "#dc2626", "🚨"),
        "high": ("#ffedd5", "#ea580c", "⚠️"),
        "medium": ("#fef3c7", "#d97706", "⚡"),
        "low": ("#e0f2fe", "#0284c7", "ℹ️"),
        "info": ("#f1f5f9", "#64748b", "📝"),
    }
    
    bg_color, border_color, icon = severity_colors.get(severity, severity_colors["info"])
    
    st.markdown(f"""
    <div style="background-color: {bg_color}; border-left: 4px solid {border_color}; 
                padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.5rem;">
        <div style="font-weight: 600;">{icon} {severity.upper()}: {issue_type}</div>
        <div style="margin-top: 0.5rem;">{message}</div>
        {f'<div style="margin-top: 0.5rem; color: #64748b; font-size: 0.875rem;">💡 {suggestion}</div>' if suggestion else ''}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Sidebar
# =============================================================================


def render_sidebar():
    """Rendert die Sidebar mit Navigation und Status."""
    with st.sidebar:
        st.markdown('<p class="main-header">🛡️ History Guardian</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">GraphRAG Faktenprüfung</p>', unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🔍 Faktenprüfung", "📥 Daten-Ingestion", "📚 Autoritative Quellen", 
             "📊 Analyse-Dashboard", "⚙️ Einstellungen"],
            label_visibility="collapsed",
        )
        
        st.divider()
        
        # System Status
        st.subheader("System Status")
        
        try:
            health = run_async(client.health_check())
            if health.get("neo4j_connected"):
                st.success("🟢 Neo4j verbunden")
            else:
                st.error("🔴 Neo4j nicht erreichbar")
        except Exception:
            st.error("🔴 API nicht erreichbar")
        
        # Stats
        try:
            stats_response = run_async(client.get_stats())
            if stats_response.get("success"):
                stats = stats_response.get("statistics", {})
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Facts", stats.get("Fact", 0))
                with col2:
                    st.metric("Claims", stats.get("Claim", 0))
        except Exception:
            pass
        
        st.divider()
        
        # Info
        st.caption("Version 0.3.0")
        st.caption("© 2024 History Guardian")
        
        return page


# =============================================================================
# Pages
# =============================================================================


def page_fact_checking():
    """Seite für Faktenprüfung."""
    st.header("🔍 Faktenprüfung")
    st.markdown("Geben Sie eine historische Behauptung ein, um sie gegen autoritative Quellen zu prüfen.")
    
    # Input
    claim_text = st.text_area(
        "Historische Behauptung",
        height=150,
        placeholder="Beispiel: Napoleon Bonaparte wurde am 15. August 1769 auf Korsika geboren und starb am 5. Mai 1821 auf St. Helena.",
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        analyze_btn = st.button("🔍 Analysieren & Prüfen", type="primary", use_container_width=True)
    with col2:
        detailed = st.checkbox("Detailanalyse", value=True)
    with col3:
        save_claim = st.checkbox("Als Claim speichern")
    
    if analyze_btn and claim_text:
        with st.spinner("Extrahiere Wissen aus Text..."):
            try:
                # Step 1: Extract
                extract_result = run_async(client.extract(claim_text))
                
                if not extract_result.get("success"):
                    st.error("Extraktion fehlgeschlagen")
                    return
                
                extraction = extract_result.get("extraction", {})
                nodes = extraction.get("nodes", [])
                relationships = extraction.get("relationships", [])
                
                st.success(f"✓ {len(nodes)} Entitäten und {len(relationships)} Beziehungen extrahiert")
                
                # Step 2: Validate
                with st.spinner("Prüfe gegen autoritative Quellen..."):
                    validation_result = run_async(client.validate(extraction))
                
                # Step 3: ML Confidence
                with st.spinner("Berechne ML-Konfidenz..."):
                    confidence_result = run_async(client.score_confidence(extraction, claim_text))
                
                # Results
                st.divider()
                
                # Main Result Cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Gesamtbewertung")
                    status = validation_result.get("overall_status", "unverifiable")
                    render_status_badge(status)
                    st.markdown(f"**{validation_result.get('summary', 'Keine Zusammenfassung')}**")
                
                with col2:
                    st.subheader("ML-Konfidenz")
                    render_confidence_gauge(confidence_result.get("confidence", 0.5))
                
                with col3:
                    st.subheader("Empfehlung")
                    st.info(validation_result.get("recommendation", "Keine Empfehlung"))
                
                if detailed:
                    st.divider()
                    
                    # Tabs for detailed info
                    tab1, tab2, tab3, tab4 = st.tabs(["📌 Entitäten", "🔗 Beziehungen", "⚠️ Probleme", "📊 Details"])
                    
                    with tab1:
                        if nodes:
                            for node in nodes:
                                render_entity_card(node)
                        else:
                            st.info("Keine Entitäten extrahiert")
                    
                    with tab2:
                        if relationships:
                            for rel in relationships:
                                st.markdown(f"**{rel.get('source_name')}** → _{rel.get('relation_type')}_ → **{rel.get('target_name')}**")
                        else:
                            st.info("Keine Beziehungen extrahiert")
                    
                    with tab3:
                        issues = validation_result.get("all_issues", [])
                        if issues:
                            for issue in issues:
                                render_issue(issue)
                        else:
                            st.success("✓ Keine Probleme gefunden")
                    
                    with tab4:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.json(confidence_result.get("components", {}))
                        with col2:
                            if confidence_result.get("explanation"):
                                st.json(confidence_result["explanation"])
                
                # Save if requested
                if save_claim:
                    ingest_result = run_async(client.ingest(claim_text, as_fact=False))
                    if ingest_result.get("success"):
                        st.success("✓ Als Claim gespeichert")
                
            except Exception as e:
                st.error(f"Fehler: {str(e)}")


def page_data_ingestion():
    """Seite für Daten-Ingestion."""
    st.header("📥 Daten-Ingestion")
    st.markdown("Fügen Sie verifizierte historische Fakten zur Ground Truth hinzu.")
    
    tab1, tab2 = st.tabs(["Einzeltext", "Batch-Upload"])
    
    with tab1:
        fact_text = st.text_area(
            "Historischer Fakt (verifiziert)",
            height=150,
            placeholder="Geben Sie einen verifizierten historischen Fakt ein...",
        )
        
        col1, col2 = st.columns(2)
        with col1:
            source_type = st.selectbox(
                "Quellentyp",
                ["Akademische Quelle", "Bibliotheks-Normdatei", "Archiv-Dokument", "Peer-reviewed"],
            )
        with col2:
            source_ref = st.text_input("Quellenreferenz", placeholder="z.B. ISBN, GND-ID, DOI...")
        
        if st.button("✓ Als Fakt speichern", type="primary"):
            if fact_text:
                with st.spinner("Speichere..."):
                    result = run_async(client.ingest(fact_text, as_fact=True))
                    if result.get("success"):
                        st.success(f"✓ {result.get('nodes_added', 0)} Entitäten als Fakten gespeichert")
                    else:
                        st.error("Speichern fehlgeschlagen")
    
    with tab2:
        uploaded_file = st.file_uploader(
            "CSV oder JSON hochladen",
            type=["csv", "json", "jsonl"],
        )
        
        if uploaded_file:
            st.info(f"Datei: {uploaded_file.name} ({uploaded_file.size} Bytes)")
            
            if st.button("🚀 Batch-Import starten"):
                st.warning("Batch-Import wird in Hintergrund gestartet...")
                # TODO: Implement batch import


def page_authority_sources():
    """Seite für autoritative Datenquellen."""
    st.header("📚 Autoritative Quellen")
    st.markdown("""
    Importieren Sie verifizierte Daten aus **bibliografischen Normdateien** und 
    **wissenschaftlichen Autoritätsdatenbanken** - keine crowdsourced Quellen.
    """)
    
    # Source Selection
    st.subheader("Verfügbare Quellen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🇩🇪 GND - Gemeinsame Normdatei</h4>
            <p>Deutsche Nationalbibliothek</p>
            <span class="source-badge">Höchste Qualität</span>
        </div>
        """, unsafe_allow_html=True)
        use_gnd = st.checkbox("GND aktivieren", value=True)
        
        st.markdown("""
        <div class="metric-card" style="margin-top: 1rem;">
            <h4>🌍 VIAF</h4>
            <p>Virtual International Authority File</p>
            <span class="source-badge">International</span>
        </div>
        """, unsafe_allow_html=True)
        use_viaf = st.checkbox("VIAF aktivieren", value=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🇺🇸 LOC Authority</h4>
            <p>Library of Congress</p>
            <span class="source-badge">US Standard</span>
        </div>
        """, unsafe_allow_html=True)
        use_loc = st.checkbox("LOC aktivieren", value=True)
        
        st.markdown("""
        <div class="metric-card" style="margin-top: 1rem;">
            <h4>🎨 Getty Vocabularies</h4>
            <p>TGN, ULAN, AAT</p>
            <span class="source-badge">Kunst & Kultur</span>
        </div>
        """, unsafe_allow_html=True)
        use_getty = st.checkbox("Getty aktivieren", value=False)
    
    st.divider()
    
    # Import Interface
    st.subheader("Daten importieren")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input(
            "Suchbegriff",
            placeholder="z.B. Napoleon Bonaparte, Französische Revolution...",
        )
    
    with col2:
        entity_type = st.selectbox(
            "Entitätstyp",
            ["Person", "Event", "Location", "Organization"],
        )
    
    if st.button("🔍 In Autoritätsdaten suchen", type="primary"):
        if search_query:
            sources = []
            if use_gnd:
                sources.append("gnd")
            if use_viaf:
                sources.append("viaf")
            if use_loc:
                sources.append("loc")
            if use_getty:
                sources.append("getty")
            
            with st.spinner(f"Suche in {len(sources)} Quellen..."):
                try:
                    result = run_async(client.import_from_authority(
                        search_query, entity_type.lower(), sources
                    ))
                    
                    if result.get("success"):
                        st.success(f"""
                        ✓ Import erfolgreich:
                        - {result.get('nodes_imported', 0)} Entitäten
                        - {result.get('relationships_imported', 0)} Beziehungen
                        - Quellen: {', '.join(result.get('sources_used', []))}
                        """)
                    else:
                        st.warning("Keine Ergebnisse gefunden")
                except Exception as e:
                    st.error(f"Import fehlgeschlagen: {e}")


def page_analysis_dashboard():
    """Analyse-Dashboard."""
    st.header("📊 Analyse-Dashboard")
    
    try:
        stats_response = run_async(client.get_stats())
        stats = stats_response.get("statistics", {})
        
        # Overview Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Gesamte Fakten", stats.get("Fact", 0), delta=None)
        with col2:
            st.metric("Geprüfte Claims", stats.get("Claim", 0))
        with col3:
            st.metric("Personen", stats.get("Person", 0))
        with col4:
            st.metric("Events", stats.get("Event", 0))
        
        st.divider()
        
        # Detailed Stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Entitäten nach Typ")
            entity_data = {
                "Person": stats.get("Person", 0),
                "Event": stats.get("Event", 0),
                "Location": stats.get("Location", 0),
                "Organization": stats.get("Organization", 0),
                "Date": stats.get("Date", 0),
            }
            st.bar_chart(entity_data)
        
        with col2:
            st.subheader("Datenqualität")
            st.json(stats)
        
    except Exception as e:
        st.error(f"Dashboard konnte nicht geladen werden: {e}")


def page_settings():
    """Einstellungen."""
    st.header("⚙️ Einstellungen")
    
    st.subheader("API Konfiguration")
    api_url = st.text_input("API URL", value=API_BASE)
    
    st.subheader("Extraktions-Einstellungen")
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Min. Konfidenz für Fakten", 0.0, 1.0, 0.8)
    with col2:
        st.slider("Max. Entitäten pro Extraktion", 1, 50, 20)
    
    st.subheader("ML-Modell")
    st.selectbox("Scoring-Modell", ["Ensemble (Standard)", "Nur Regelbasiert", "Nur ML"])
    
    st.divider()
    
    st.subheader("Gefahrenzone")
    if st.button("🗑️ Alle Claims löschen", type="secondary"):
        if st.checkbox("Ich bin sicher"):
            st.warning("Claims werden gelöscht...")


# =============================================================================
# Main
# =============================================================================


def main():
    """Hauptfunktion."""
    page = render_sidebar()
    
    if page == "🔍 Faktenprüfung":
        page_fact_checking()
    elif page == "📥 Daten-Ingestion":
        page_data_ingestion()
    elif page == "📚 Autoritative Quellen":
        page_authority_sources()
    elif page == "📊 Analyse-Dashboard":
        page_analysis_dashboard()
    elif page == "⚙️ Einstellungen":
        page_settings()


if __name__ == "__main__":
    main()
