"""
Equinor Investor Relations — Gradio E-Reader Frontend

Layout:
  Topbar → document picker + page navigation (always visible)
  Main   → PDF viewer (left 58%) | Chat (right 42%)
  Below  → Extracted tables accordion
"""

import base64
import re
import sys
import traceback
import json
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
import pandas as pd
import pdfplumber

sys.path.insert(0, str(Path(__file__).parent / "src"))
from investor_relations_scraper import config

RAW_DIR       = config.RAW_DIR
PROCESSED_DIR = config.PROCESSED_DIR

# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def get_pdf_list() -> List[str]:
    return sorted(f.name for f in RAW_DIR.glob("*.pdf"))


def _map_tables_to_pages(pdf_path: Path, base_name: str) -> Dict[int, List[pd.DataFrame]]:
    mapping: Dict[int, List[pd.DataFrame]] = {}
    new_pat = re.compile(re.escape(base_name) + r"_table_p(\d+)_\d+\.csv")
    old_pat = re.compile(re.escape(base_name) + r"_table_\d+\.csv")
    for csv_path in sorted(PROCESSED_DIR.glob(f"{base_name}_table_*.csv")):
        m = new_pat.match(csv_path.name)
        if m:
            pn = int(m.group(1))
        elif old_pat.match(csv_path.name):
            pn = 0
        else:
            continue
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                mapping.setdefault(pn, []).append(df)
        except Exception:
            pass
    return mapping


def _render_page_image(pdf_path: Path, page_num: int):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if 1 <= page_num <= len(pdf.pages):
                return pdf.pages[page_num - 1].to_image(resolution=180).original
    except Exception as e:
        print(f"Error rendering page: {e}")
    return None


def _count_pages(pdf_path: Path) -> int:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


def _page_to_base64(pdf_path: Path, page_num: int) -> Optional[str]:
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=150, first_page=page_num, last_page=page_num)
        if not images:
            return None
        buf = BytesIO()
        images[0].save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"Error converting page: {e}")
        return None

# ---------------------------------------------------------------------------
# Doc state cache
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
        self.pdf_name = pdf_name
        self.total_pages = _count_pages(pdf_path)
        self.table_map = _map_tables_to_pages(pdf_path, pdf_path.stem)
        print(f"📂 Loaded {pdf_name}: {self.total_pages} pages")

doc_state = DocState()
_qa_engine = None

# ---------------------------------------------------------------------------
# Page callbacks
# ---------------------------------------------------------------------------

def _build_page_outputs(pdf_name: str, page_num: int):
    pdf_path = RAW_DIR / pdf_name
    total = doc_state.total_pages
    img = _render_page_image(pdf_path, page_num)
    page_tables = doc_state.table_map.get(page_num, [])
    table_df = page_tables[0] if page_tables else None
    status = f"Page {page_num} of {total}  ·  {pdf_name}"
    # Return the actual PDF path for the download button
    dl_path = str(pdf_path) if pdf_path.exists() else None
    return (
        gr.Slider(value=page_num, maximum=max(total, 1)),
        img,
        table_df,
        gr.Accordion(visible=table_df is not None),
        status,
        dl_path,
    )


def on_doc_selected(pdf_name: str):
    if not pdf_name:
        return gr.Slider(value=1, maximum=1), None, None, gr.Accordion(visible=False), "No document selected", None
    doc_state.load(pdf_name)
    return _build_page_outputs(pdf_name, 1)


def on_page_change(page_num, pdf_name: str):
    if not pdf_name:
        return gr.Slider(), None, None, gr.Accordion(visible=False), "Select a document first", None
    doc_state.load(pdf_name)
    page_num = max(1, min(int(page_num), doc_state.total_pages or 1))
    return _build_page_outputs(pdf_name, page_num)


def on_prev_page(page_num, pdf_name):
    return on_page_change(max(1, int(page_num) - 1), pdf_name)

def on_next_page(page_num, pdf_name):
    return on_page_change(min(doc_state.total_pages or 1, int(page_num) + 1), pdf_name)

# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

def _parse_json_blocks(content: str) -> List[pd.DataFrame]:
    """Parse JSON format into DataFrames."""
    content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
    try:
        data = json.loads(content)
        if not isinstance(data, list):
            return []
        tables = []
        for table_obj in data:
            if isinstance(table_obj, dict) and "headers" in table_obj and "rows" in table_obj:
                df = pd.DataFrame(table_obj["rows"], columns=table_obj["headers"])
                # pad rows if GPT missed some columns
                if not df.empty:
                    tables.append(df)
        return tables
    except Exception as e:
        print(f"Failed to parse JSON blocks: {e}\nContent was:\n{content[:200]}...")
        return []



def _gpt_extract(client, model: str, data_uri: str, page_num: int) -> List[pd.DataFrame]:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                {"type": "text", "text": (
                    "Extract ALL tables from this financial report page as a specific JSON format.\n"
                    "CRITICAL RULES FOR ACCURACY:\n"
                    "1. DO NOT calculate, guess, or hallucinate numbers. Copy EXACTLY what you see.\n"
                    "2. If a cell is blank in the image, strictly output an empty string `\"\"`.\n"
                    "3. If a cell contains a dash `-`, output a dash `\"-\"`.\n"
                    "4. Preserve all numbers exactly with commas (e.g., if it says 10,620, output \"10,620\").\n"
                    "5. Convert parenthesised negatives: (75) MUST become \"-75\".\n"
                    "6. Remove footnote markers (* ** 1 2 etc.).\n"
                    "7. Use exact column headers. Flatten multi-level headers with spaces.\n"
                    "8. Keep row labels in the first column. Repeat merged cell labels on each row they span.\n\n"
                    "FORMATTING REQUIREMENTS:\n"
                    "You MUST output ONLY a valid JSON array of table objects. Example format:\n"
                    "[\n"
                    "  {\n"
                    "    \"headers\": [\"Indicator\", \"1Q25\", \"1Q24\"],\n"
                    "    \"rows\": [\n"
                    "      [\"Earnings\", \"10,620\", \"9,800\"],\n"
                    "      [\"Taxes\", \"-3,226\", \"-3,849\"]\n"
                    "    ]\n"
                    "  }\n"
                    "]\n\n"
                    "- Return ONLY JSON code. No markdown fences (```), no explanations.\n"
                    "- If NO tables exist on the page, respond exactly: NO_TABLES"
                )}
            ]}],
            max_tokens=4096, temperature=0.0,
            response_format={"type": "json_object"} if "gpt-4o" not in model.lower() else None # Force JSON if supported
        )
        content = r.choices[0].message.content.strip()
        if content == "NO_TABLES" or '"headers"' not in content:
            return []
        return _parse_json_blocks(content)
    except Exception as e:
        print(f"  ✗ Extract error p{page_num}: {e}")
        return []


def _gpt_verify(client, model: str, data_uri: str, tables: List[pd.DataFrame], page_num: int) -> List[pd.DataFrame]:
    if not tables:
        return tables
    json_payload = []
    for df in tables:
        json_payload.append({
            "headers": df.columns.tolist(),
            "rows": df.values.tolist()
        })
    json_text = json.dumps(json_payload, indent=2)
    
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                {"type": "text", "text": (
                    f"Below is a first-pass JSON extraction from this financial report page.\n"
                    f"CRITICAL TASK: Check it carefully against the original image and correct ANY hallucinated, calculated, or incorrect numbers.\n\n"
                    f"Current extraction:\n{json_text}\n\n"
                    "RULES FOR CORRECTION:\n"
                    "1. EVERY number must EXACTLY match the image. Do not calculate missing totals.\n"
                    "2. Fix wrong column placements and misaligned rows.\n"
                    "3. (100) must be \"-100\".\n"
                    "4. Empty cells must be empty strings `\"\"`.\n"
                    "5. Output the corrected JSON only (using same array of dicts format).\n"
                    "6. You MUST return exactly the same number of tables.\n"
                    "7. Return ONLY raw JSON code, no markdown fences or preambles."
                )}
            ]}],
            max_tokens=4096, temperature=0.0,
        )
        content = r.choices[0].message.content.strip()
        if not content or content == "NO_TABLES" or "headers" not in content:
            return tables
        corrected = _parse_json_blocks(content)
        if len(corrected) == len(tables):
            print(f"  ✅ Verified {len(tables)} table(s) on page {page_num}")
            return corrected
        print(f"  ⚠ Verify mismatch ({len(corrected)} vs {len(tables)}), keeping original")
        return tables
    except Exception as e:
        print(f"  ⚠ Verify error p{page_num}: {e}")
        return tables


def on_extract_page_table(pdf_name: str, page_num: int):
    if not pdf_name:
        return None, gr.Group(visible=False), "⚠ No document selected", {}, 0, gr.Dropdown(choices=[])
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.get_openai_api_key())
        pdf_path = RAW_DIR / pdf_name
        pn = int(page_num)
        print(f"  🖼 Rendering page {pn}…")
        uri = _page_to_base64(pdf_path, pn)
        if not uri:
            return None, gr.Group(visible=False), f"⚠ Could not render page {pn}", {}, 0, gr.Dropdown(choices=[])
        print("  🔍 Pass 1 — extracting…")
        tables = _gpt_extract(client, config.MODEL_TABLE_EXTRACTOR, uri, pn)
        if not tables:
            return None, gr.Group(visible=False), "ℹ No tables found on this page.", {}, 0, gr.Dropdown(choices=[])
        print("  🔄 Pass 2 — verifying…")
        tables = _gpt_verify(client, config.MODEL_TABLE_EXTRACTOR, uri, tables, pn)
        state = {"tables": [df.to_dict(orient="records") for df in tables],
                 "columns": [list(df.columns) for df in tables]}
        n = len(tables)
        first_df = tables[0]
        return (
            first_df,
            gr.Group(visible=True),
            f"✅ Extracted & verified {n} table(s). Reviewing 1 of {n}.",
            state, 0,
            gr.Dropdown(choices=list(first_df.columns), value=None, label="Select column to delete"),
        )
    except Exception as e:
        traceback.print_exc()
        return None, gr.Group(visible=False), f"❌ {str(e)}", {}, 0, gr.Dropdown(choices=[])


def _show_table_at(state, idx):
    if not state or not state.get("tables"):
        return "", 0, "No tables loaded.", gr.Dropdown(choices=[])
    tables = state["tables"]
    columns = state["columns"]
    idx = max(0, min(idx, len(tables) - 1))
    df = pd.DataFrame(tables[idx], columns=columns[idx])
    n = len(tables)
    status = f"Reviewing table {idx + 1} of {n}."
    col_choices = gr.Dropdown(choices=columns[idx], value=None, label="Select column to delete")
    return df, idx, status, col_choices
def on_prev_table(state, idx):  return _show_table_at(state, idx - 1)
def on_next_table(state, idx):  return _show_table_at(state, idx + 1)


def on_delete_column(df: pd.DataFrame, col_to_delete: str, state: dict, idx: int):
    """Remove a column from the currently displayed table."""
    if df is None or not col_to_delete:
        return df, gr.Dropdown(choices=[]), state

    if hasattr(df, 'columns') and col_to_delete in df.columns:
        df = df.drop(columns=[col_to_delete])

    # Update state
    if state and state.get("tables"):
        state["tables"][idx] = df.to_dict(orient="records")
        state["columns"][idx] = list(df.columns)

    remaining_cols = list(df.columns) if hasattr(df, 'columns') else []
    return df, gr.Dropdown(choices=remaining_cols, value=None, label="Select column to delete"), state


def on_discard_table(state: dict, idx: int):
    if not state or not state.get("tables"):
        return "", gr.Group(visible=False), "No tables.", state, 0, gr.Dropdown(choices=[])
    tables = state["tables"]
    columns = state["columns"]
    tables.pop(idx)
    columns.pop(idx)
    new_state = {"tables": tables, "columns": columns}
    if not tables:
        return "", gr.Group(visible=False), "ℹ All tables discarded.", new_state, 0, gr.Dropdown(choices=[])
    new_idx = min(idx, len(tables) - 1)
    df, new_idx, status, col_dd = _show_table_at(new_state, new_idx)
    return df, gr.Group(visible=True), "🗑 Discarded. " + status, new_state, new_idx, col_dd


def on_save_extracted_table(df: pd.DataFrame, pdf_name: str, page_num: int,
                            state: dict, idx: int):
    if df is None or (hasattr(df, 'empty') and df.empty):
        return "⚠ No data to save.", None, gr.Group(visible=False), state, idx, gr.Dropdown(choices=[])
    try:
        base = Path(pdf_name).stem
        pn = int(page_num)
        csv_file = PROCESSED_DIR / f"{base}_table_p{pn}_{idx+1}.csv"
        df.to_csv(csv_file, index=False)
        doc_state.table_map.setdefault(pn, [])
        slot = idx
        if slot < len(doc_state.table_map[pn]):
            doc_state.table_map[pn][slot] = df
        else:
            doc_state.table_map[pn].append(df)
        if _qa_engine and hasattr(_qa_engine, 'db_manager'):
            _qa_engine.db_manager.reload_csv(csv_file.name)
        msg = f"✅ Saved {csv_file.name}! "
    except Exception as e:
        return f"❌ {str(e)}", "", gr.Group(visible=False), state, idx, gr.Dropdown(choices=[])
    
    tables = state["tables"]
    columns = state["columns"]
    tables.pop(idx)
    columns.pop(idx)
    new_state = {"tables": tables, "columns": columns}

    if not tables:
        return msg + "All tables reviewed!", None, gr.Group(visible=False), new_state, 0, gr.Dropdown(choices=[])
    new_idx = min(idx, len(tables) - 1)
    next_df, new_idx, nav_status, col_dd = _show_table_at(new_state, new_idx)
    return msg + nav_status, next_df, gr.Group(visible=True), new_state, new_idx, col_dd

# ---------------------------------------------------------------------------
# Chat callbacks
# ---------------------------------------------------------------------------

def on_chat_submit(user_message: str, history: list):
    if not user_message.strip():
        return history, ""
    return history + [{"role": "user", "content": user_message}], ""


def on_chat_respond(history: list, refs_state: list):
    if not history:
        return history, refs_state, gr.Accordion(open=False)
    last = history[-1]
    if isinstance(last, dict):
        if last.get("role") != "user":
            return history, refs_state, gr.Accordion(open=False)
        user_message = last["content"]
    else:
        user_message = str(last[0]) if last else ""
    print(f"💬 Q: {user_message}")
    try:
        result   = _qa_engine.answer_question(user_message)
        answer   = result.get("answer", "Sorry, I couldn't generate an answer.")
        src_refs = result.get("source_refs", [])
        sources  = result.get("sources", [])
        if sources:
            answer += "\n\n---\n**📚 Sources:** " + " · ".join(f"`{s}`" for s in sources)
    except Exception as e:
        traceback.print_exc()
        answer = f"Error: {e}"; src_refs = []
    history.append({"role": "assistant", "content": answer})
    return history, src_refs, gr.Accordion(open=bool(src_refs))


def on_clear_chat():
    if _qa_engine: _qa_engine.clear_conversation()
    return []

# ---------------------------------------------------------------------------
# CSS  — large PDF, black chat text, clean Apple reading style
# ---------------------------------------------------------------------------

CSS = """
/* ── Tokens ── */
:root {
    --bg:        #f2f2f7;
    --card:      #ffffff;
    --border:    rgba(0,0,0,0.08);
    --text:      #1c1c1e;
    --sub:       #6c6c70;
    --accent:    #007aff;
    --accent-h:  #005ecb;
    --danger:    #ff3b30;
    --radius:    12px;
    --radius-sm: 8px;
    --shadow-sm: 0 1px 4px rgba(0,0,0,0.06), 0 2px 12px rgba(0,0,0,0.04);
    --shadow:    0 2px 8px rgba(0,0,0,0.07), 0 8px 24px rgba(0,0,0,0.06);
    --font:      -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Helvetica Neue", sans-serif;
}
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--font) !important;
    color: var(--text) !important;
}
footer { display:none !important; }

/* ── Topbar ── */
#topbar {
    background: rgba(255,255,255,0.82);
    backdrop-filter: saturate(200%) blur(28px);
    -webkit-backdrop-filter: saturate(200%) blur(28px);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0;
    padding: 0 24px;
    height: 56px;
    margin: -8px -8px 20px;
    position: sticky; top: 0; z-index: 200;
}
#topbar-brand {
    font-size: 15px; font-weight: 700; letter-spacing: -0.4px;
    color: var(--text); white-space: nowrap; margin-right: 24px;
}
#topbar-brand span { color: var(--accent); }

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
}

/* ── PDF viewer — fill column width ── */
#pdf-card { padding: 0; overflow: hidden; }
#pdf-card img {
    display: block !important;
    width: 100% !important;
    height: auto !important;
    border-radius: var(--radius) !important;
    object-fit: contain !important;
}
/* Remove Gradio image chrome */
#pdf-card .image-frame, #pdf-card [data-testid="image"] {
    border: none !important; background: transparent !important;
    box-shadow: none !important;
}
#pdf-card .image-container { padding: 0 !important; }

/* ── Section label pill ── */
.sec {
    font-size: 10px; font-weight: 700; letter-spacing: 0.9px;
    text-transform: uppercase; color: var(--sub);
    margin: 0 0 8px 2px;
}

/* ── Chat — FORCE BLACK TEXT ── */
#chat-panel .message-wrap p,
#chat-panel .message-wrap span,
#chat-panel .message-wrap li,
#chat-panel .message-wrap code,
#chat-panel [data-testid="bot"] *,
#chat-panel [data-testid="user"] *,
#chat-panel .gr-chatbot *,
.gr-chatbot .message p,
.gr-chatbot .message span,
.gr-chatbot .prose p,
.gr-chatbot .prose span,
.gr-chatbot .prose li,
.gr-chatbot .prose strong,
.gr-chatbot .prose em {
    color: #1c1c1e !important;
}
/* User bubble */
#chat-panel .message.user, #chat-panel [data-testid="user"] {
    background: #e8f0fe !important;
    border-radius: 14px 14px 4px 14px !important;
    color: #1c1c1e !important;
}
/* Bot bubble */
#chat-panel .message.bot, #chat-panel [data-testid="bot"] {
    background: #f5f5f7 !important;
    border-radius: 14px 14px 14px 4px !important;
    color: #1c1c1e !important;
}
/* Chat container */
#chat-panel .gr-chatbot, #chat-panel [data-testid="chatbot"] {
    background: transparent !important;
    border: none !important; box-shadow: none !important;
}

/* ── Buttons ── */
.btn-primary, button.primary, .gr-button-primary {
    background: var(--accent) !important; color: #fff !important;
    border: none !important; border-radius: var(--radius-sm) !important;
    font-weight: 500 !important; font-family: var(--font) !important;
    font-size: 14px !important; transition: background 0.13s !important;
}
.btn-primary:hover { background: var(--accent-h) !important; }
.gr-button-secondary {
    background: rgba(0,122,255,0.09) !important; color: var(--accent) !important;
    border: none !important; border-radius: var(--radius-sm) !important;
    font-family: var(--font) !important; font-weight: 500 !important;
}
button[data-testid*="stop"], .gr-button-stop {
    background: rgba(255,59,48,0.07) !important; color: var(--danger) !important;
    border: none !important; border-radius: var(--radius-sm) !important;
    font-family: var(--font) !important;
}

/* Narrow nav buttons */
#nav-btns button { min-width: 44px !important; padding: 6px 10px !important; }

/* ── Inputs ── */
textarea, input[type="text"], select {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    background: #fff !important;
    font-family: var(--font) !important;
    font-size: 14px !important;
    color: var(--text) !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,122,255,0.13) !important;
    outline: none !important;
}

/* ── Labels ── */
label > span, .form > label {
    font-size: 11.5px !important; font-weight: 500 !important;
    color: var(--sub) !important; font-family: var(--font) !important;
    letter-spacing: 0.1px !important;
}

/* ── Invisible status box ── */
#status-box > label { display: none !important; }
#status-box textarea {
    background: transparent !important; border: none !important;
    box-shadow: none !important; resize: none !important;
    font-size: 12px !important; color: var(--sub) !important;
    padding: 2px 0 !important; min-height: 0 !important;
}

/* ── Dataframe ── */
.gr-dataframe {
    border-radius: var(--radius-sm) !important;
    font-family: var(--font) !important;
    font-size: 13px !important;
}

/* ── Accordion ── */
.gr-accordion {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border) !important;
    background: var(--card) !important;
}
"""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Equinor IR Explorer",
) as demo:

    # ── Sticky topbar ──
    gr.HTML("""
    <div id="topbar">
        <div id="topbar-brand">📊 Equinor <span>IR</span> Explorer</div>
    </div>
    """)

    # ── Shared state ──
    source_refs_state      = gr.State([])
    extracted_tables_state = gr.State({})
    table_idx_state        = gr.State(0)

    # ── Document controls row ──
    with gr.Row():
        pdf_dropdown = gr.Dropdown(
            choices=get_pdf_list(), label="Document", scale=4,
        )
        prev_btn = gr.Button("◀", scale=0, min_width=44)
        page_slider = gr.Slider(minimum=1, maximum=1, step=1, value=1, label="Page", scale=3)
        next_btn = gr.Button("▶", scale=0, min_width=44)
        dl_btn = gr.DownloadButton("⬇ PDF", label="Download PDF", scale=1, min_width=100, visible=False)

    status_box = gr.Textbox(
        label="", interactive=False, value="Select a document to begin",
        elem_id="status-box", lines=1,
    )

    # ── Main 2-column body ──
    with gr.Row(equal_height=False):

        # ════════════════ LEFT — PDF viewer (wider) ════════════════
        with gr.Column(scale=11):

            gr.HTML('<p class="sec">Document Viewer</p>')
            with gr.Group(elem_classes=["card"], elem_id="pdf-card"):
                page_image = gr.Image(label="", type="pil", show_label=False)

            # Saved tables for this page
            table_group = gr.Accordion("📊 Saved Table for this Page", open=True, visible=False)
            with table_group:
                extracted_table = gr.Dataframe(label="", interactive=False, wrap=True)

            # Extract panel
            with gr.Accordion("🔍 Extract & Verify Tables (This Page)", open=True):
                with gr.Row():
                    extract_btn = gr.Button(
                        "Extract Tables with GPT-4o-mini",
                        variant="secondary", scale=2,
                    )
                extract_status = gr.Textbox(
                    label="", interactive=False, value="",
                    lines=1, show_label=False,
                )

                extract_group = gr.Group(visible=False)
                with extract_group:
                    gr.HTML('<p class="sec" style="margin-top:8px">Review Extracted Table</p>')
                    editable_table = gr.Dataframe(label="", interactive=True, wrap=True)

                    with gr.Row():
                        col_to_delete = gr.Dropdown(
                            choices=[], value=None,
                            label="Column to delete", scale=3,
                        )
                        delete_col_btn = gr.Button("🗑 Delete Column", variant="stop", scale=1, min_width=130)

                    with gr.Row():
                        prev_table_btn    = gr.Button("◀ Prev",    scale=1, min_width=80)
                        save_table_btn    = gr.Button("💾 Save",   variant="primary", scale=2)
                        discard_table_btn = gr.Button("Discard",   variant="stop", scale=1, min_width=90)
                        next_table_btn    = gr.Button("Next ▶",    scale=1, min_width=80)

        # ════════════════ RIGHT — Chat ════════════════
        with gr.Column(scale=7, elem_id="chat-panel"):

            gr.HTML('<p class="sec">AI Assistant</p>')
            with gr.Group(elem_classes=["card"]):
                chatbot = gr.Chatbot(
                    label="", height=520, show_label=False,
                    placeholder="*Ask anything about the reports…*",
                )

                with gr.Accordion("📖 Source References", open=False) as sources_accordion:
                    sources_info = gr.Markdown("*Ask a question to see page-level references.*")
                    source_ref_dataset = gr.Dataset(
                        components=["text"], label="", visible=False, samples=[],
                    )

                with gr.Row():
                    chat_input = gr.Textbox(
                        label="", placeholder="Ask a question…",
                        scale=5, show_label=False,
                    )
                    chat_send  = gr.Button("Send",  variant="primary", scale=1, min_width=70)
                    chat_clear = gr.Button("Clear", scale=1, min_width=60)

            gr.Examples(
                examples=[
                    "What was the adjusted operating income for Q4 2024?",
                    "Summarize the production highlights from Q3 2025.",
                    "Compare European gas prices across all quarters of 2025.",
                    "What is Equinor's capital distribution guidance?",
                ],
                inputs=chat_input, label="Examples",
            )

    # ── Wiring: Document ──

    shared_outputs = [page_slider, page_image, extracted_table, table_group, status_box, dl_btn]

    pdf_dropdown.change(fn=on_doc_selected,  inputs=[pdf_dropdown],              outputs=shared_outputs)
    page_slider.release(fn=on_page_change,   inputs=[page_slider, pdf_dropdown], outputs=shared_outputs)
    prev_btn.click(     fn=on_prev_page,      inputs=[page_slider, pdf_dropdown], outputs=shared_outputs)
    next_btn.click(     fn=on_next_page,      inputs=[page_slider, pdf_dropdown], outputs=shared_outputs)

    # ── Wiring: Table Extraction ──

    extract_btn.click(
        fn=lambda: gr.Textbox(value="⏳ Pass 1 — extracting…  (Pass 2 verification follows)"),
        outputs=[extract_status],
    ).then(
        fn=on_extract_page_table,
        inputs=[pdf_dropdown, page_slider],
        outputs=[editable_table, extract_group, extract_status, extracted_tables_state, table_idx_state, col_to_delete],
    )

    delete_col_btn.click(
        fn=on_delete_column,
        inputs=[editable_table, col_to_delete, extracted_tables_state, table_idx_state],
        outputs=[editable_table, col_to_delete, extracted_tables_state],
    )
    save_table_btn.click(
        fn=on_save_extracted_table,
        inputs=[editable_table, pdf_dropdown, page_slider, extracted_tables_state, table_idx_state],
        outputs=[extract_status, editable_table, extract_group, extracted_tables_state, table_idx_state, col_to_delete],
    ).then(fn=on_page_change, inputs=[page_slider, pdf_dropdown], outputs=shared_outputs)

    discard_table_btn.click(
        fn=on_discard_table,
        inputs=[extracted_tables_state, table_idx_state],
        outputs=[editable_table, extract_group, extract_status, extracted_tables_state, table_idx_state, col_to_delete],
    )
    prev_table_btn.click(fn=on_prev_table, inputs=[extracted_tables_state, table_idx_state],
                         outputs=[editable_table, table_idx_state, extract_status, col_to_delete])
    next_table_btn.click(fn=on_next_table, inputs=[extracted_tables_state, table_idx_state],
                         outputs=[editable_table, table_idx_state, extract_status, col_to_delete])

    # ── Wiring: Chat ──

    def render_refs(refs):
        if not refs: return "*No source references for this answer.*"
        return "\n\n".join(
            f"📄 **{r.get('pdf','')}**" +
            (f" · page {r['page']}" if r.get('page') else "") +
            f" — *{r.get('title', r.get('pdf',''))}*"
            for r in refs
        )

    def chat_respond_full(history, refs_state):
        history, new_refs, accordion = on_chat_respond(history, refs_state)
        samples = [[f"📄 {r['pdf']} · pg {r.get('page','?')} — {r.get('title','')}"] for r in new_refs]
        return history, new_refs, accordion, render_refs(new_refs), gr.Dataset(samples=samples, visible=bool(new_refs))

    for trigger in [chat_send.click, chat_input.submit]:
        trigger(
            fn=on_chat_submit, inputs=[chat_input, chatbot], outputs=[chatbot, chat_input],
        ).then(
            fn=chat_respond_full,
            inputs=[chatbot, source_refs_state],
            outputs=[chatbot, source_refs_state, sources_accordion, sources_info, source_ref_dataset],
        )

    def on_source_select(refs, evt: gr.SelectData):
        try:
            ref = refs[evt.index]
            return (ref.get("pdf",""),) + tuple(on_page_change(ref.get("page") or 1, ref.get("pdf","")))
        except Exception:
            return (gr.update(),) * (1 + len(shared_outputs))

    source_ref_dataset.select(fn=on_source_select, inputs=[source_refs_state],
                               outputs=[pdf_dropdown] + shared_outputs)
    chat_clear.click(fn=on_clear_chat, outputs=[chatbot])

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config.ensure_directories()
    from investor_relations_scraper.qa_engine import QAEngine
    print("🚀 Initializing QA Engine…")
    _qa_engine = QAEngine(data_dir=str(PROCESSED_DIR))
    _qa_engine.load_and_index()
    print("✅ QA Engine ready.")
    my_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "Helvetica Neue", "sans-serif"],
    )
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS, theme=my_theme)
