"""
Equinor Investor Relations — Gradio E-Reader Frontend

Three-column layout:
  Left   → Extracted text + tables (from the extractor pipeline)
  Center → PDF page rendering (pdfplumber)
  Right  → Per-page notes (persisted to JSON)
  Bottom → AI Q&A chat (using the QAEngine, initialized lazily)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import pdfplumber



# Ensure src/ is importable when running from root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from investor_relations_scraper import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NOTES_PATH = config.DATA_DIR / "notes.json"
RAW_DIR = config.RAW_DIR
PROCESSED_DIR = config.PROCESSED_DIR

# ---------------------------------------------------------------------------
# Notes persistence
# ---------------------------------------------------------------------------

def _load_all_notes() -> dict:
    if NOTES_PATH.exists():
        return json.loads(NOTES_PATH.read_text(encoding="utf-8"))
    return {}


def _save_all_notes(notes: dict):
    NOTES_PATH.write_text(
        json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _notes_key(pdf_name: str, page: int) -> str:
    return f"{pdf_name}:{page}"


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def get_pdf_list() -> List[str]:
    return sorted(f.name for f in RAW_DIR.glob("*.pdf"))


def _map_tables_to_pages(
    pdf_path: Path, base_name: str
) -> Dict[int, List[pd.DataFrame]]:
    """Map tables to pages using the page number embedded in CSV filenames.
    
    Expects filenames like: {base_name}_table_p{page}_{idx}.csv
    Falls back to old format {base_name}_table_{idx}.csv (no page mapping).
    """
    import re
    mapping: Dict[int, List[pd.DataFrame]] = {}
    
    # Pattern: base_table_p{page}_{idx}.csv  (new format with page number)
    new_pattern = re.compile(
        re.escape(base_name) + r"_table_p(\d+)_\d+\.csv"
    )
    # Pattern: base_table_{idx}.csv  (old format without page number)
    old_pattern = re.compile(
        re.escape(base_name) + r"_table_\d+\.csv"
    )
    
    for csv_path in sorted(PROCESSED_DIR.glob(f"{base_name}_table_*.csv")):
        m = new_pattern.match(csv_path.name)
        if m:
            page_num = int(m.group(1))
        elif old_pattern.match(csv_path.name):
            # Old format — assign to page 0 (will show on all pages or none)
            page_num = 0
        else:
            continue
        
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                mapping.setdefault(page_num, []).append(df)
        except Exception:
            pass
    
    return mapping




def _render_page_image(pdf_path: Path, page_num: int):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if 1 <= page_num <= len(pdf.pages):
                return pdf.pages[page_num - 1].to_image(resolution=150).original
    except Exception as e:
        print(f"Error rendering page: {e}")
    return None


def _count_pages(pdf_path: Path) -> int:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Per-document cache
# ---------------------------------------------------------------------------

class DocState:
    def __init__(self):
        self.pdf_name: Optional[str] = None
        self.total_pages: int = 0
        self.table_map: Dict[int, List[pd.DataFrame]] = {}

    def load(self, pdf_name: str):
        if self.pdf_name == pdf_name:
            return
        pdf_path = RAW_DIR / pdf_name
        base_name = pdf_path.stem
        self.pdf_name = pdf_name
        self.total_pages = _count_pages(pdf_path)
        self.table_map = _map_tables_to_pages(pdf_path, base_name)
        tables_total = sum(len(v) for v in self.table_map.values())
        print(f"📂 Loaded {pdf_name}: {self.total_pages} pages, {tables_total} tables")


doc_state = DocState()

# ---------------------------------------------------------------------------
# QA Engine (initialized at startup)
# ---------------------------------------------------------------------------

_qa_engine = None


# ---------------------------------------------------------------------------
# Callbacks — Document Inspector
# ---------------------------------------------------------------------------

def on_doc_selected(pdf_name: str):
    if not pdf_name:
        return (
            gr.Slider(value=1, maximum=1),
            None, None, gr.Group(visible=False), "",
            "No document selected",
        )
    doc_state.load(pdf_name)
    return _build_page_outputs(pdf_name, 1)


def on_page_change(page_num, pdf_name: str):
    if not pdf_name:
        return (
            gr.Slider(),
            None, None, gr.Group(visible=False), "",
            "Select a document first",
        )
    doc_state.load(pdf_name)
    page_num = max(1, min(int(page_num), doc_state.total_pages or 1))
    return _build_page_outputs(pdf_name, page_num)


def _build_page_outputs(pdf_name: str, page_num: int):
    pdf_path = RAW_DIR / pdf_name
    total = doc_state.total_pages

    img = _render_page_image(pdf_path, page_num)

    page_tables = doc_state.table_map.get(page_num, [])
    if page_tables:
        table_df = page_tables[0]
        table_visible = True
    else:
        table_df = None
        table_visible = False

    notes = _load_all_notes()
    note_text = notes.get(_notes_key(pdf_name, page_num), "")
    status = f"Page {page_num} / {total}  —  {pdf_name}"

    return (
        gr.Slider(value=page_num, maximum=max(total, 1)),
        img, table_df,
        gr.Group(visible=table_visible),
        note_text, status,
    )


def on_save_note(pdf_name: str, page_num, note_text: str):
    if not pdf_name:
        return "⚠ No document selected"
    notes = _load_all_notes()
    key = _notes_key(pdf_name, int(page_num))
    if note_text.strip():
        notes[key] = note_text
    else:
        notes.pop(key, None)
    _save_all_notes(notes)
    return "✅ Note saved"


def on_prev_page(page_num, pdf_name):
    return on_page_change(max(1, int(page_num) - 1), pdf_name)


def on_next_page(page_num, pdf_name):
    total = doc_state.total_pages or 1
    return on_page_change(min(total, int(page_num) + 1), pdf_name)


# ---------------------------------------------------------------------------
# Callbacks — Interactive Table Extraction (multi-table navigation)
# ---------------------------------------------------------------------------

def on_extract_page_table(pdf_name: str, page_num: int):
    """Call GPT-4o-mini vision to extract ALL tables from the current page."""
    if not pdf_name:
        return None, gr.Group(visible=False), "\u26a0 No document selected", [], 0

    try:
        from investor_relations_scraper.extractors import GPT4VisionExtractor
        extractor = GPT4VisionExtractor(model=config.MODEL_TABLE_EXTRACTOR)
        pdf_path = RAW_DIR / pdf_name

        images = extractor._pdf_pages_to_images(pdf_path)

        page_uri = None
        for p, uri in images:
            if p == int(page_num):
                page_uri = uri
                break

        if not page_uri:
            return None, gr.Group(visible=False), f"\u26a0 Could not render page {page_num}", [], 0

        tables = extractor._extract_tables_from_image(page_uri, int(page_num))

        if not tables:
            return None, gr.Group(visible=False), "\u2139 No tables found on this page.", [], 0

        # Convert DataFrames to JSON-serialisable dicts for gr.State
        tables_json = [df.to_dict(orient="records") for df in tables]
        columns_json = [list(df.columns) for df in tables]
        state = {"tables": tables_json, "columns": columns_json}

        n = len(tables)
        first_df = tables[0]
        status = f"\u2705 Extracted {n} table(s). Reviewing table 1 of {n}. Edit below, then save or discard."
        return first_df, gr.Group(visible=True), status, state, 0

    except Exception as e:
        return None, gr.Group(visible=False), f"\u274c Extraction error: {str(e)}", [], 0


def _show_table_at(state: dict, idx: int):
    """Return the DataFrame at index idx from state, plus updated status."""
    if not state or not state.get("tables"):
        return None, 0, "No tables loaded."
    tables = state["tables"]
    columns = state["columns"]
    idx = max(0, min(idx, len(tables) - 1))
    df = pd.DataFrame(tables[idx], columns=columns[idx])
    n = len(tables)
    status = f"Reviewing table {idx + 1} of {n}. Edit below, then save or discard."
    return df, idx, status


def on_prev_table(state: dict, idx: int):
    """Navigate to the previous extracted table."""
    df, new_idx, status = _show_table_at(state, idx - 1)
    return df, new_idx, status


def on_next_table(state: dict, idx: int):
    """Navigate to the next extracted table."""
    df, new_idx, status = _show_table_at(state, idx + 1)
    return df, new_idx, status


def on_discard_table(state: dict, idx: int):
    """Discard the current table and move to the next (or close if none left)."""
    if not state or not state.get("tables"):
        return None, gr.Group(visible=False), "No tables.", state, 0

    tables = state["tables"]
    columns = state["columns"]
    # Remove the current table
    tables.pop(idx)
    columns.pop(idx)
    new_state = {"tables": tables, "columns": columns}

    if not tables:
        return None, gr.Group(visible=False), "\u2139 All tables discarded.", new_state, 0

    new_idx = min(idx, len(tables) - 1)
    df, new_idx, status = _show_table_at(new_state, new_idx)
    status = "\ud83d\uddd1 Discarded. " + status
    return df, gr.Group(visible=True), status, new_state, new_idx

def on_save_extracted_table(df: pd.DataFrame, pdf_name: str, page_num: int,
                            state: dict, idx: int):
    """Save the currently displayed table, remove it from state, advance to next."""
    if df is None or (hasattr(df, 'empty') and df.empty):
        return "\u26a0 No data to save.", None, gr.Group(visible=False), state, idx

    try:
        base_name = Path(pdf_name).stem
        page_num_int = int(page_num)

        # Use the table's position in state (idx+1 within page) to keep filenames unique
        csv_idx = idx + 1
        csv_file = PROCESSED_DIR / f"{base_name}_table_p{page_num_int}_{csv_idx}.csv"
        df.to_csv(csv_file, index=False)

        doc_state.table_map.setdefault(page_num_int, [])
        # Replace or append at this slot
        if csv_idx - 1 < len(doc_state.table_map[page_num_int]):
            doc_state.table_map[page_num_int][csv_idx - 1] = df
        else:
            doc_state.table_map[page_num_int].append(df)

        if _qa_engine and hasattr(_qa_engine, 'db_manager'):
            _qa_engine.db_manager.reload_csv(csv_file.name)

        msg = f"\u2705 Saved {csv_file.name}! "
    except Exception as e:
        return f"\u274c Save error: {str(e)}", None, gr.Group(visible=False), state, idx

    # After saving, discard this table from state and advance
    tables = state.get("tables", [])
    columns = state.get("columns", [])
    if tables:
        tables.pop(idx)
        columns.pop(idx)
    new_state = {"tables": tables, "columns": columns}

    if not tables:
        return msg + "All tables reviewed!", None, gr.Group(visible=False), new_state, 0

    new_idx = min(idx, len(tables) - 1)
    next_df, new_idx, nav_status = _show_table_at(new_state, new_idx)
    return msg + nav_status, next_df, gr.Group(visible=True), new_state, new_idx

# ---------------------------------------------------------------------------
# Callbacks — Chat (manual Chatbot, avoids gr.ChatInterface schema bug)
# ---------------------------------------------------------------------------

def on_chat_submit(user_message: str, history: list):
    """Process a chat message through the QA engine."""
    if not user_message.strip():
        return history, ""

    history = history + [{"role": "user", "content": user_message}]
    return history, ""


def on_chat_respond(history: list, source_refs_state: list):
    """Generate AI response (runs after user msg is displayed)."""
    if not history or history[-1]["role"] != "user":
        return history, source_refs_state, gr.Accordion(open=False)

    user_message = history[-1]["content"]
    print(f"💬 Q: {user_message}")

    try:
        result = _qa_engine.answer_question(user_message)
        answer = result.get("answer", "Sorry, I couldn't generate an answer.")
        source_refs = result.get("source_refs", [])
        sources = result.get("sources", [])

        # Append inline source list to the answer for quick reference
        if sources:
            answer += "\n\n---\n**📚 Sources:** " + " · ".join(
                f"`{s}`" for s in sources
            )
    except Exception as e:
        answer = f"Error: {e}"
        source_refs = []

    history.append({"role": "assistant", "content": answer})
    open_sources = len(source_refs) > 0
    return history, source_refs, gr.Accordion(open=open_sources)


def on_clear_chat():
    _qa_engine.clear_conversation()
    return []


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
#right-col { border-left:  1px solid #e0e0e0; }
footer     { display: none !important; }
"""

with gr.Blocks(
    title="Equinor IR Explorer",
) as demo:

    # ── Header ──
    gr.Markdown(
        "# 📊 Equinor Investor Relations Explorer\n"
        "Browse reports page-by-page · view tables · "
        "take notes · ask questions"
    )

    # ── Top bar: document picker + navigation ──
    with gr.Row():
        pdf_dropdown = gr.Dropdown(
            choices=get_pdf_list(),
            label="📄 Select Document",
            scale=3,
        )
        prev_btn = gr.Button("◀ Prev", scale=1, min_width=80)
        page_slider = gr.Slider(
            minimum=1, maximum=1, step=1, value=1,
            label="Page", scale=2,
        )
        next_btn = gr.Button("Next ▶", scale=1, min_width=80)
        status_box = gr.Textbox(
            label="Status", interactive=False, scale=2,
            value="Select a document to begin",
        )

    # ── Per-page extraction state ──
    extracted_tables_state = gr.State({})   # {"tables": [...], "columns": [...]}
    table_idx_state = gr.State(0)           # index of currently displayed table

    # ── Two-column layout ──
    with gr.Row(equal_height=True):

        # LEFT — PDF page image + tables
        with gr.Column(scale=3):
            gr.Markdown("### 🖼️ Original PDF Page")
            page_image = gr.Image(
                label="Page render", type="pil",
            )
            
            # Interactive Ollama Extraction
            with gr.Row():
                extract_btn = gr.Button("🔍 Extract Tables with GPT-4o-mini (This Page)", variant="secondary")

            extract_status = gr.Textbox(label="Extraction Status", interactive=False, visible=False)

            extract_group = gr.Group(visible=False)
            with extract_group:
                with gr.Row():
                    gr.Markdown("### ✏️ Edit Extracted Table")
                    table_counter = gr.Markdown("")
                editable_table = gr.Dataframe(
                    label="Editable CSV", interactive=True, wrap=True,
                )
                with gr.Row():
                    prev_table_btn = gr.Button("◀ Prev Table", scale=1)
                    save_table_btn = gr.Button("💾 Save", variant="primary", scale=2)
                    discard_table_btn = gr.Button("🗑 Discard", variant="stop", scale=1)
                    next_table_btn = gr.Button("Next Table ▶", scale=1)
            
            table_group = gr.Group(visible=False)
            with table_group:
                gr.Markdown("### 📊 Existing Tables for this Page")
                extracted_table = gr.Dataframe(
                    label="Cleaned CSV", interactive=False, wrap=True,
                )

        # RIGHT — notes
        with gr.Column(scale=1, elem_id="right-col"):
            gr.Markdown("### 🗒️ Notes")
            note_area = gr.Textbox(
                label="Your notes for this page",
                lines=18, max_lines=30,
                placeholder="Type your notes here…",
            )
            save_note_btn = gr.Button("💾 Save Note", variant="primary")
            note_status = gr.Textbox(label="", interactive=False)

    # ── Bottom: AI Chat ──
    gr.Markdown("---")
    gr.Markdown(
        "### 💬 AI Assistant  — Ask anything about the reports\n"
        "*The QA engine loads on first question (one-time, may take a few minutes).*"
    )

    chatbot = gr.Chatbot(label="Chat", height=350)

    # Source references panel (auto-opens after each answered question)
    source_refs_state = gr.State([])  # list of {pdf, page, title} dicts
    with gr.Accordion("📖 Source References — click a row to jump to that page", open=False) as sources_accordion:
        sources_info = gr.Markdown("*Ask a question to see page-level source references here.*")
        source_ref_dataset = gr.Dataset(
            components=["text"],
            label="",
            visible=False,
            samples=[],
        )

    with gr.Row():
        chat_input = gr.Textbox(
            label="Your question",
            placeholder="e.g. What was the adjusted operating income for Q4 2024?",
            scale=5,
        )
        chat_send = gr.Button("Send", variant="primary", scale=1)
        chat_clear = gr.Button("🗑️ Clear", scale=1)

    gr.Examples(
        examples=[
            "What was the adjusted operating income for Q4 2024?",
            "Summarize the production highlights from Q3 2025.",
            "Compare European gas prices across all quarters of 2025.",
            "What is Equinor's capital distribution guidance?",
        ],
        inputs=chat_input,
    )

    # ── Wiring: Document Inspector ──

    shared_outputs = [
        page_slider, page_image,
        extracted_table, table_group, note_area, status_box,
    ]

    pdf_dropdown.change(
        fn=on_doc_selected, inputs=[pdf_dropdown], outputs=shared_outputs,
    )
    page_slider.release(
        fn=on_page_change, inputs=[page_slider, pdf_dropdown],
        outputs=shared_outputs,
    )
    prev_btn.click(
        fn=on_prev_page, inputs=[page_slider, pdf_dropdown],
        outputs=shared_outputs,
    )
    next_btn.click(
        fn=on_next_page, inputs=[page_slider, pdf_dropdown],
        outputs=shared_outputs,
    )
    save_note_btn.click(
        fn=on_save_note, inputs=[pdf_dropdown, page_slider, note_area],
        outputs=[note_status],
    )
    
    # ── Wiring: Interactive Table Extraction ──
    
    extract_btn.click(
        fn=lambda: gr.Textbox(visible=True, value="⏳ Extracting... this may take a moment."),
        outputs=[extract_status]
    ).then(
        fn=on_extract_page_table,
        inputs=[pdf_dropdown, page_slider],
        outputs=[editable_table, extract_group, extract_status, extracted_tables_state, table_idx_state]
    )

    save_table_btn.click(
        fn=on_save_extracted_table,
        inputs=[editable_table, pdf_dropdown, page_slider, extracted_tables_state, table_idx_state],
        outputs=[extract_status, editable_table, extract_group, extracted_tables_state, table_idx_state]
    ).then(
        fn=on_page_change,
        inputs=[page_slider, pdf_dropdown],
        outputs=shared_outputs
    )

    discard_table_btn.click(
        fn=on_discard_table,
        inputs=[extracted_tables_state, table_idx_state],
        outputs=[editable_table, extract_group, extract_status, extracted_tables_state, table_idx_state]
    )

    prev_table_btn.click(
        fn=on_prev_table,
        inputs=[extracted_tables_state, table_idx_state],
        outputs=[editable_table, table_idx_state, extract_status]
    )

    next_table_btn.click(
        fn=on_next_table,
        inputs=[extracted_tables_state, table_idx_state],
        outputs=[editable_table, table_idx_state, extract_status]
    )

    # ── Wiring: Chat ──

    def render_source_refs(refs: list) -> str:
        """Convert source_refs list to a markdown string with clickable links."""
        if not refs:
            return "*No source references for this answer.*"
        lines = []
        for r in refs:
            pdf = r.get("pdf", "")
            page = r.get("page")
            title = r.get("title", pdf)
            page_str = f" · page {page}" if page else ""
            # Show as a styled badge — clicking updates doc viewer via another handler
            lines.append(f"📄 **{pdf}**{page_str} — *{title}*")
        return "\n\n".join(lines)

    def navigate_to_source(refs: list, evt: gr.SelectData):
        """Navigate the document viewer to a clicked source reference."""
        try:
            ref = refs[evt.index]
            pdf = ref.get("pdf", "")
            page = ref.get("page") or 1
            # Reuse existing on_page_change to navigate
            result = on_page_change(page, pdf)
            return (pdf,) + tuple(result)
        except Exception as e:
            return (gr.update(),) * (1 + len(shared_outputs))


    def chat_respond_with_render(history, refs_state):
        history, new_refs, accordion = on_chat_respond(history, refs_state)
        md = render_source_refs(new_refs)
        samples = [[f"📄 {r['pdf']} · pg {r['page'] or '?'} — {r['title']}"] for r in new_refs]
        visible = len(new_refs) > 0
        return history, new_refs, accordion, md, gr.Dataset(samples=samples, visible=visible)

    chat_send.click(
        fn=on_chat_submit,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input],
    ).then(
        fn=chat_respond_with_render,
        inputs=[chatbot, source_refs_state],
        outputs=[chatbot, source_refs_state, sources_accordion, sources_info, source_ref_dataset],
    )

    chat_input.submit(
        fn=on_chat_submit,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input],
    ).then(
        fn=chat_respond_with_render,
        inputs=[chatbot, source_refs_state],
        outputs=[chatbot, source_refs_state, sources_accordion, sources_info, source_ref_dataset],
    )

    def on_source_select(refs, evt: gr.SelectData):
        """Navigate to the selected source page when a row is clicked."""
        try:
            ref = refs[evt.index]
            pdf = ref.get("pdf", "")
            page = ref.get("page") or 1
            page_results = on_page_change(page, pdf)
            return (pdf,) + tuple(page_results)
        except Exception:
            return (gr.update(),) * (1 + len(shared_outputs))

    source_ref_dataset.select(
        fn=on_source_select,
        inputs=[source_refs_state],
        outputs=[pdf_dropdown] + shared_outputs,
    )

    chat_clear.click(fn=on_clear_chat, outputs=[chatbot])


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config.ensure_directories()

    # Pre-load QA Engine at startup (uses cached index if files haven't changed)
    from investor_relations_scraper.qa_engine import QAEngine
    print("🚀 Initializing QA Engine at startup…")
    _qa_engine = QAEngine(data_dir=str(PROCESSED_DIR))
    _qa_engine.load_and_index()
    print("✅ QA Engine ready — launching server.")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.slate,
        ),
        css=CSS,
    )

