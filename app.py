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

# ---------------------------------------------------------------------------
# Monkey-patch: fix gradio_client bug with pydantic v2 on Python 3.9
# pydantic v2 emits additionalProperties: True (a bool), but
# gradio_client.utils.get_type() does `"const" in schema` which fails
# when schema is a bool.  We patch _json_schema_to_python_type to guard.
# ---------------------------------------------------------------------------
import gradio_client.utils as _gc_utils

_orig_json_schema = _gc_utils._json_schema_to_python_type

def _patched_json_schema(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_json_schema(schema, defs)

_gc_utils._json_schema_to_python_type = _patched_json_schema

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
            gr.update(value=1, maximum=1),
            None, None, gr.update(visible=False), "",
            "No document selected",
        )
    doc_state.load(pdf_name)
    return _build_page_outputs(pdf_name, 1)


def on_page_change(page_num, pdf_name: str):
    if not pdf_name:
        return (
            gr.update(),
            None, None, gr.update(visible=False), "",
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
        gr.update(value=page_num, maximum=max(total, 1)),
        img, table_df,
        gr.update(visible=table_visible),
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
# Callbacks — Chat (manual Chatbot, avoids gr.ChatInterface schema bug)
# ---------------------------------------------------------------------------

def on_chat_submit(user_message: str, history: list):
    """Process a chat message through the QA engine."""
    if not user_message.strip():
        return history, ""

    history = history + [[user_message, None]]
    return history, ""


def on_chat_respond(history: list):
    """Generate AI response (runs after user msg is displayed)."""
    if not history or history[-1][1] is not None:
        return history

    user_message = history[-1][0]
    print(f"💬 Q: {user_message}")

    try:
        result = _qa_engine.answer_question(user_message)
        answer = result.get("answer", "Sorry, I couldn't generate an answer.")
        sources = result.get("sources", [])
        if sources:
            answer += "\n\n**Sources:** " + ", ".join(sources)
    except Exception as e:
        answer = f"Error: {e}"

    history[-1][1] = answer
    return history


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
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
    ),
    css=CSS,
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

    # ── Two-column layout ──
    with gr.Row(equal_height=True):

        # LEFT — PDF page image + tables
        with gr.Column(scale=3):
            gr.Markdown("### 🖼️ Original PDF Page")
            page_image = gr.Image(
                label="Page render", type="pil",
                show_download_button=True,
            )
            table_group = gr.Group(visible=False)
            with table_group:
                gr.Markdown("### 📊 Extracted Table")
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

    # ── Bottom: AI Chat (manual Chatbot to avoid schema bug) ──
    gr.Markdown("---")
    gr.Markdown(
        "### 💬 AI Assistant  — Ask anything about the reports\n"
        "*The QA engine loads on first question (one-time, may take a few minutes).*"
    )

    chatbot = gr.Chatbot(label="Chat", height=350)
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

    # ── Wiring: Chat ──

    chat_send.click(
        fn=on_chat_submit,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input],
    ).then(
        fn=on_chat_respond,
        inputs=[chatbot],
        outputs=[chatbot],
    )

    chat_input.submit(
        fn=on_chat_submit,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input],
    ).then(
        fn=on_chat_respond,
        inputs=[chatbot],
        outputs=[chatbot],
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

    demo.launch(server_name="0.0.0.0", server_port=7860)

