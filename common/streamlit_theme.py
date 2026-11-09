"""Shared Streamlit look. Call `apply()` once at the top of the app after st.set_page_config."""
import streamlit as st
CSS = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
html, body, [class*="css"] { font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif; }
h1, h2, h3 { font-family: "Space Grotesk", "Helvetica Neue", Arial, sans-serif !important; letter-spacing: .01em; }
code, pre, .stCode { font-family: "IBM Plex Mono", Menlo, monospace !important; }
.stApp { background: #0c1018; color: #e8ecf6; }
section[data-testid="stSidebar"] { background: #121826; border-right: 1px solid rgba(226,232,245,.12); }
div[data-testid="stMetric"] { background: #182033; border: 1px solid rgba(226,232,245,.12); border-radius: 12px; padding: 12px 16px; }
.stButton > button { background: #4fd1c5; color: #0c1018; border: 0; border-radius: 8px; font-weight: 600; }
.stButton > button:hover { background: #6fe3d6; color: #0c1018; }
div[data-testid="stDataFrame"], .stAlert { border-radius: 12px; }
.badge { font-family: "IBM Plex Mono", monospace; font-size: .75rem; padding: 3px 9px; border: 1px solid #4fd1c5; color: #4fd1c5; border-radius: 999px; }
</style>
"""
def apply(service: str | None = None, subtitle: str | None = None, badge: str | None = None):
    st.markdown(CSS, unsafe_allow_html=True)
    if service:
        b = f' <span class="badge">{badge}</span>' if badge else ""
        st.markdown(f"## {service}{b}", unsafe_allow_html=True)
        if subtitle: st.caption(subtitle)
