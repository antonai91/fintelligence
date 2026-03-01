# Quick Reference

## Setup

```bash
uv sync
cp .env.example .env
# Edit .env with your OpenAI API Key
```

## Basic Commands

### Interactive UI

```bash
uv run python app.py
```

* **Upload PDF**: Use the button in the top bar to add and index new documents on the fly.

### Extractor (Process PDFs)

```bash
# Process all downloaded PDFs
uv run ir-extractor

# Process a single specific file
uv run ir-extractor --file "Q1-2025-report.pdf"

# Skip cleaning (faster, raw output only)
uv run ir-extractor --no-cleaning
```

### QA Chat

```bash
uv run python examples/chat_qa.py
```

## Extraction Methods

By default, the system uses **pdfplumber** for fast text extraction. Tables are extracted on demand in the Gradio UI using **GPT-4o Vision** (OpenAI); extracted tables are synced to DuckDB for QA and Text-to-SQL.

## Common Options

### Extractor

| Option | Description |
| --- | --- |
| `--file FILENAME` | Process specific file |
| `--skip-text` | Skip text extraction |
| `--skip-tables` | Skip table extraction |
| `--no-cleaning` | Skip OpenAI text cleaning |
| `--raw-dir DIR` | Custom input directory |
| `--processed-dir DIR` | Custom output directory |

## Output Files

| Output | Description |
| --- | --- |
| `{name}_text.txt` | Clean text content |
| `{name}_table_p{page}_{n}.csv` | Extracted tables in CSV format (page optional in name); synced to DuckDB for QA |

## Table Storage (DuckDB)

Extracted table CSVs are ingested into `data/vector_db/tables.duckdb`. The QA engine can run Text-to-SQL over these tables. Edits to tables in the Gradio UI are saved back to CSV and reloaded into DuckDB.

## Metadata Filtering

The QA Engine automatically extracts and uses:

* **Company**: Tesla, Equinor, etc.
* **Year/Quarter**: 2024, Q1, etc.
* **Type**: Transcript, Report, etc.

## Troubleshooting

### OpenAI API Error

```bash
# Verify API key in .env
cat .env
```

### File Not Found

Ensure the filename passed to `--file` exists in `data/raw/`.

## Performance Tips

* **Text Only**: The CLI extracts text by default (no table extraction in batch). Use `--skip-tables` if you run with table extraction enabled. Tables are usually extracted on demand in the Gradio UI.
* **No Cleaning**: Use `--no-cleaning` to skip OpenAI processing for raw extraction.
* **Process Single File**: Always test with one file first before running the full batch.

## Documentation

* [**System Architecture**](architecture.md): High-level system design.
* [**README**](../README.md): Main project guide.
