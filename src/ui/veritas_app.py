"""
Veritas UI v0.7.1 - Streamlit Frontend

Seiten:
- Start: Uebersicht
- Faktencheck: Claim pruefen
- Knowledge Base: Statistiken
- Mining: Fact Mining
"""

import streamlit as st
import requests
from typing import Optional, Dict, Any

# =============================================================================
# Config
# =============================================================================

st.set_page_config(
    page_title="Veritas",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"


# =============================================================================
# Styles
# =============================================================================

st.markdown(
    """
<style>
/* Dark Theme */
.stApp { background-color: #0e1117; }

/* Verdict Cards */
.verdict-card {
    padding: 24px;
    border-radius: 16px;
    margin: 16px 0;
    text-align: center;
}
.verdict-true { background: linear-gradient(135deg, #1a3d1a, #2d5a2d); border: 2px solid #4caf50; }
.verdict-false { background: linear-gradient(135deg, #3d1a1a, #5a2d2d); border: 2px solid #f44336; }
.verdict-label { font-size: 2rem; font-weight: bold; }

/* Info Card */
.info-card {
    background: #1e1f20;
    border: 1px solid #3c4043;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
}

/* Correction Box */
.correction-box {
    background: #1a3d1a;
    border: 1px solid #4caf50;
    border-radius: 12px;
    padding: 16px;
    margin: 16px 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Session State
# =============================================================================

if "page" not in st.session_state:
    st.session_state.page = "home"
if "claim" not in st.session_state:
    st.session_state.claim = ""


# =============================================================================
# API Helpers
# =============================================================================


def api_get(endpoint: str) -> Optional[Dict]:
    """GET Request an API."""
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=30)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
    return None


def api_post(endpoint: str, data: Dict = None) -> Optional[Dict]:
    """POST Request an API."""
    try:
        response = requests.post(f"{API_URL}{endpoint}", json=data or {}, timeout=120)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
    return None


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.title("Veritas")
    st.caption("Self-Improving Fact Checker v0.7.1")

    st.divider()

    if st.button(
        "Start",
        use_container_width=True,
        type="primary" if st.session_state.page == "home" else "secondary",
    ):
        st.session_state.page = "home"
        st.rerun()

    if st.button(
        "Faktencheck",
        use_container_width=True,
        type="primary" if st.session_state.page == "check" else "secondary",
    ):
        st.session_state.page = "check"
        st.rerun()

    st.divider()
    st.caption("Self-Learning")

    if st.button(
        "Knowledge Base",
        use_container_width=True,
        type="primary" if st.session_state.page == "kb" else "secondary",
    ):
        st.session_state.page = "kb"
        st.rerun()

    if st.button(
        "Mining",
        use_container_width=True,
        type="primary" if st.session_state.page == "mining" else "secondary",
    ):
        st.session_state.page = "mining"
        st.rerun()

    st.divider()

    # API Status
    health = api_get("/health")
    if health and health.get("status") == "healthy":
        st.success("API Online")
    else:
        st.error("API Offline")
        st.code("uvicorn src.api.main:app --port 8000")


# =============================================================================
# Pages
# =============================================================================

# HOME
if st.session_state.page == "home":
    st.title("Veritas")
    st.subheader("Self-Improving Fact Checker")

    st.markdown(
        """
    Pruefe Behauptungen gegen autoritative Quellen:
    
    - **Wikidata** - Strukturierte Daten
    - **Wikipedia** - Enzyklopaedie  
    - **LLM** - Sprachmodell (Groq)
    - **Local KB** - Eigene Knowledge Base
    """
    )

    st.divider()

    # Quick Check
    st.subheader("Schnellcheck")
    claim = st.text_input(
        "Behauptung eingeben:", placeholder="z.B. Deutschland liegt in Europa"
    )

    if st.button("Pruefen", type="primary", use_container_width=True) and claim:
        st.session_state.claim = claim
        st.session_state.page = "result"
        st.rerun()

    # Stats
    st.divider()
    stats = api_get("/learning/facts/stats")
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Fakten in KB", stats.get("total_facts", 0))
        c2.metric("TRUE", stats.get("by_verdict", {}).get("true", 0))
        c3.metric("FALSE", stats.get("by_verdict", {}).get("false", 0))


# FACT CHECK
elif st.session_state.page == "check":
    st.title("Faktencheck")

    claim = st.text_area(
        "Behauptung:", height=100, placeholder="Gib eine Behauptung ein..."
    )

    if st.button("Pruefen", type="primary", use_container_width=True) and claim:
        st.session_state.claim = claim
        st.session_state.page = "result"
        st.rerun()


# RESULT
elif st.session_state.page == "result":
    st.title("Ergebnis")

    claim = st.session_state.claim
    st.info(f'**Behauptung:** "{claim}"')

    with st.spinner("Pruefe Fakten..."):
        result = api_post("/factcheck/check", {"claim": claim})

    if result:
        verdict = result.get("verdict", "")
        label = result.get("verdict_label", "")
        confidence = result.get("confidence", 0)

        # Verdict Card
        css_class = "verdict-true" if verdict == "true" else "verdict-false"
        st.markdown(
            f"""
        <div class="verdict-card {css_class}">
            <div class="verdict-label">{label}</div>
            <div style="font-size: 1.2rem; margin-top: 8px;">
                Konfidenz: {confidence:.0%}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Explanation
        st.markdown(f"**Erklaerung:** {result.get('explanation', '')}")

        # Correction
        if result.get("correction"):
            st.markdown(
                f"""
            <div class="correction-box">
                <strong>Richtig:</strong> {result['correction']}
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Meta
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Typ", result.get("claim_type", "-"))
        c2.metric("Zeit", f"{result.get('processing_time_ms', 0)}ms")
        c3.metric("Quellen", result.get("sources_checked", 0))
        c4.metric("Cache", "Ja" if result.get("cached") else "Nein")

        # Sources
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LLM", "Ja" if result.get("llm_used") else "Nein")
        c2.metric("Wikidata", "Ja" if result.get("wikidata_used") else "Nein")
        c3.metric("Wikipedia", "Ja" if result.get("wikipedia_used") else "Nein")
        c4.metric("Local KB", "Ja" if result.get("local_kb_used") else "Nein")

        # Evidence
        if result.get("evidence"):
            with st.expander("Evidenz anzeigen"):
                for e in result["evidence"]:
                    icon = "[+]" if e.get("supports_claim") else "[-]"
                    color = "#4caf50" if e.get("supports_claim") else "#f44336"
                    st.markdown(
                        f"""
                    <div class="info-card">
                        <strong style="color: {color};">{icon} {e.get('source', '')}</strong>
                        ({e.get('confidence', 0):.0%})<br>
                        <span style="color: #9aa0a6;">{e.get('content', '')[:200]}</span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
    else:
        st.error("API nicht erreichbar")

    if st.button("Neue Pruefung", use_container_width=True):
        st.session_state.page = "check"
        st.rerun()


# KNOWLEDGE BASE
elif st.session_state.page == "kb":
    st.title("Knowledge Base")
    st.caption("Verifizierte Fakten aus autoritativen Quellen")

    stats = api_get("/learning/facts/stats")

    if stats:
        # Hauptzahlen
        c1, c2, c3 = st.columns(3)
        c1.metric("Gesamt Fakten", stats.get("total_facts", 0))
        by_verdict = stats.get("by_verdict", {})
        c2.metric("TRUE", by_verdict.get("true", 0))
        c3.metric("FALSE", by_verdict.get("false", 0))

        # Nach Quelle
        st.divider()
        st.subheader("Nach Quelle")
        by_source = stats.get("by_source", {})
        if by_source:
            cols = st.columns(len(by_source))
            for i, (src, cnt) in enumerate(by_source.items()):
                cols[i].metric(src, cnt)

        # Nach Typ
        st.subheader("Nach Typ")
        by_type = stats.get("by_type", {})
        if by_type:
            cols = st.columns(len(by_type))
            for i, (typ, cnt) in enumerate(by_type.items()):
                cols[i].metric(typ, cnt)

        # Sample
        st.divider()
        st.subheader("Beispiele")
        sample = api_get("/learning/facts/sample?limit=10")
        if sample and sample.get("sample"):
            for item in sample["sample"]:
                verdict = "[TRUE]" if item.get("is_true") else "[FALSE]"
                color = "#4caf50" if item.get("is_true") else "#f44336"
                st.markdown(
                    f'<span style="color: {color};">{verdict}</span> {item.get("claim", "")}',
                    unsafe_allow_html=True,
                )
    else:
        st.warning("Keine Daten. Starte Mining um Fakten zu sammeln.")


# MINING
elif st.session_state.page == "mining":
    st.title("Fact Mining")
    st.caption("Extrahiere Fakten aus autoritativen Quellen")

    st.markdown(
        """
    Das System lernt automatisch aus:
    - **Wikidata** - Laender, Hauptstaedte, Personen
    - **Wikipedia** - Bekannte Irrtuemer
    - **Adversarial** - Generierte FALSE-Varianten
    
    **Jeder Mining-Lauf holt NEUE Daten** (Offset-Tracking).
    """
    )

    st.divider()

    # Status
    stats = api_get("/learning/facts/stats")
    if stats:
        c1, c2 = st.columns(2)
        c1.metric("Fakten in KB", stats.get("total_facts", 0))
        c2.info(f"Storage: {stats.get('storage_path', '')}")

    # Test
    st.subheader("1. Verbindungstest")
    if st.button("Wikidata testen"):
        with st.spinner("Teste Verbindung..."):
            test = api_post("/learning/test-wikidata")
        if test and test.get("success"):
            st.success(f"OK - {test.get('results_count', 0)} Ergebnisse")
            st.caption(f"Beispiel: {', '.join(test.get('sample', [])[:3])}")
        else:
            st.error(f"Fehler: {test.get('error', 'Unbekannt')}")

    st.divider()

    # Mining
    st.subheader("2. Mining starten")
    st.warning("Mining kann 1-2 Minuten dauern")

    if st.button("Mining starten", type="primary", use_container_width=True):
        with st.spinner("Extrahiere Fakten aus Wikidata, Wikipedia..."):
            result = api_post("/learning/mine")

        if result:
            if result.get("success"):
                total_new = result.get("total_new", 0)

                if total_new > 0:
                    st.success(f"Erfolgreich! {total_new} neue Fakten.")
                else:
                    st.info("Keine neuen Fakten (bereits vorhanden).")

                # Stats
                st.subheader("Ergebnis")
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Geografie", result.get("geographic", 0) + result.get("capitals", 0)
                )
                c2.metric("Personen", result.get("biographical", 0))
                c3.metric("Adversarial", result.get("adversarial", 0))

                st.caption(f"Gesamt in KB: {result.get('total_in_db', 0)}")

                # Offsets
                offsets = result.get("offsets", {})
                if offsets:
                    with st.expander("Mining-Fortschritt"):
                        st.markdown("Das System merkt sich wo es aufgehoert hat.")
                        for k, v in offsets.items():
                            if k != "misconceptions_done":
                                st.text(f"{k}: {v} verarbeitet")

                # Errors
                if result.get("errors"):
                    with st.expander(f"Fehler ({len(result['errors'])})"):
                        for e in result["errors"]:
                            st.error(e)
            else:
                st.error("Mining fehlgeschlagen")
                if result.get("errors"):
                    for e in result["errors"]:
                        st.error(e)
        else:
            st.error("API nicht erreichbar")

    st.divider()

    # Reload KB
    st.subheader("3. Knowledge Base aktualisieren")
    st.caption("Nach Mining sollte die KB im Fact Checker neu geladen werden.")

    if st.button("KB neu laden"):
        result = api_post("/learning/reload-kb")
        if result and result.get("success"):
            st.success("KB neu geladen")
        else:
            st.error("Fehler beim Laden")
