# UI standard for every service (added 2026-11-08, Max: "ensure we have a good UI and video demonstration")

One product, fourteen doors. Every interface must read as the same instrument panel:
- Dark panel, one accent (`--accent` teal), amber for a human decision or a warning, rose only for errors. Tokens and components in `common/ui.css`.
- Type: Space Grotesk for headings, IBM Plex Sans for body, IBM Plex Mono for numbers, ids and code. Load from Google Fonts (link in ui.css header comment).
- Every page has: `header.top` with the service name, a one-line "what this does", a provider or model badge where relevant, and the port. Then the primary action first (upload, question, pick a user), result beside or below it, numbers as `.stat` tiles, charts as `figure` with a caption that says what to look at.
- Empty state says what to do ("Drop a blueprint or pick a sample"). Loading shows `.spinner` and disables the button. Errors say what went wrong and what to try.
- Sample inputs are one click away on every page (sample images, sample tickets, sample emails, a default user). An interviewer must get a real result within ten seconds of opening the page.
- Phone width works (400px): grids stack, no horizontal scroll.
- HTML pages: link `/static/ui.css` (copy `common/ui.css` into the app's `static/` at setup, or serve the common folder). Streamlit: inject the same tokens with `st.markdown("<style>…</style>", unsafe_allow_html=True)` using `common/streamlit_theme.py` and set `.streamlit/config.toml` `[theme] base="dark" primaryColor="#4fd1c5" backgroundColor="#0c1018" secondaryBackgroundColor="#182033" textColor="#e8ecf6" font="sans serif"`. Gradio: `gr.themes.Base()` with the same colours via `theme.set(...)` or a `css=` string using the tokens.
- No lorem ipsum, no placeholder numbers: every number on screen comes from metrics.json or the live model.
- A `GET /health` on every service that returns `{"ok": true, "service": "<name>", "model_loaded": bool}`.

Video demonstration: after all services are up, a Playwright script records each UI performing one real action; clips are stitched into one walkthrough with title cards (coordinator owns this). Make the happy path scriptable: stable element ids on the primary input, the primary button and the result container (`#input`, `#run`, `#result`).
