"""Shared Gradio look matching common/ui.css: `gr.Blocks(theme=theme(), css=CSS)`.
Added by the portfolio CV package (headstorm_blueprint, headstorm_menu); any Gradio app can import it."""
from __future__ import annotations

import gradio as gr

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root { --bg:#0c1018; --bg-2:#121826; --panel:#182033; --line:rgba(226,232,245,.12); --fg:#e8ecf6; --fg-2:#aab3c8; --fg-3:#7c869d; --accent:#4fd1c5; --warn:#f2b84b; --danger:#f0718a; --ok:#6fe3a0; }
body, .gradio-container { background: var(--bg) !important; color: var(--fg); font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif; }
.gradio-container { max-width: 1120px !important; margin: 0 auto; padding: 0 16px 48px; }
h1, h2, h3 { font-family: "Space Grotesk", "Helvetica Neue", Arial, sans-serif; letter-spacing: .01em; }
code, pre, .cm-editor { font-family: "IBM Plex Mono", Menlo, monospace; }
header.top { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; padding:18px 0 10px; border-bottom:1px solid var(--line); margin-bottom:16px; }
header.top h1 { font-size:1.35rem; margin:0; color: var(--fg); } header.top .sub { color:var(--fg-3); font-size:.9rem; flex:1; min-width:0; }
header.top .badge { font-family:"IBM Plex Mono", monospace; font-size:.75rem; padding:3px 9px; border:1px solid var(--line); border-radius:999px; color:var(--fg-2); }
header.top .badge.on { border-color:var(--accent); color:var(--accent); }
.block, .form, .gr-box { background: var(--panel) !important; border-color: var(--line) !important; border-radius: 12px !important; }
button.primary { background: var(--accent) !important; color: var(--bg) !important; border: 0 !important; font-weight: 600; }
button.secondary { background: transparent !important; color: var(--fg) !important; border: 1px solid var(--line) !important; }
.gallery .gallery-item, .examples { border-color: var(--line) !important; }
.stat b { font-family: "Space Grotesk", sans-serif; font-size: 1.6rem; } .stat span { font-size: .8rem; color: var(--fg-3); }
@media (max-width: 480px) { .gradio-container { padding: 0 16px 32px; } }
"""


def theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e6fffb", c100="#b3f5ec", c200="#8ae9de", c300="#6fe3d6", c400="#4fd1c5", c500="#4fd1c5",
            c600="#3fb8ad", c700="#2f9a90", c800="#237a72", c900="#185a54", c950="#0f3b37",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f5f7fb", c100="#e8ecf6", c200="#aab3c8", c300="#7c869d", c400="#5a6377", c500="#3b4356",
            c600="#2a3245", c700="#182033", c800="#121826", c900="#0c1018", c950="#070a0f",
        ),
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "Helvetica Neue", "Arial", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "Menlo", "monospace"],
    ).set(
        body_background_fill="#0c1018",
        body_background_fill_dark="#0c1018",
        body_text_color="#e8ecf6",
        body_text_color_dark="#e8ecf6",
        block_background_fill="#182033",
        block_background_fill_dark="#182033",
        block_border_color="rgba(226,232,245,.12)",
        block_border_color_dark="rgba(226,232,245,.12)",
        block_label_background_fill="#121826",
        block_label_background_fill_dark="#121826",
        block_label_text_color="#aab3c8",
        block_label_text_color_dark="#aab3c8",
        block_title_text_color="#e8ecf6",
        block_title_text_color_dark="#e8ecf6",
        input_background_fill="#121826",
        input_background_fill_dark="#121826",
        button_primary_background_fill="#4fd1c5",
        button_primary_background_fill_dark="#4fd1c5",
        button_primary_background_fill_hover="#6fe3d6",
        button_primary_background_fill_hover_dark="#6fe3d6",
        button_primary_text_color="#0c1018",
        button_primary_text_color_dark="#0c1018",
        button_secondary_background_fill="#182033",
        button_secondary_background_fill_dark="#182033",
        button_secondary_text_color="#e8ecf6",
        button_secondary_text_color_dark="#e8ecf6",
        border_color_primary="rgba(226,232,245,.12)",
        border_color_primary_dark="rgba(226,232,245,.12)",
        background_fill_secondary="#121826",
        background_fill_secondary_dark="#121826",
        table_even_background_fill="#182033",
        table_even_background_fill_dark="#182033",
        table_odd_background_fill="#121826",
        table_odd_background_fill_dark="#121826",
    )

# Gradio decides light/dark from the browser; our panels are dark-only, so force the dark class on load
# (and honour ?__theme=dark, which the demo recorder appends). Pass as `gr.Blocks(..., js=DARK_JS)`.
DARK_JS = """() => { document.body.classList.add('dark'); document.documentElement.classList.add('dark'); }"""
