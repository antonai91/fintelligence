# Equinor Investor Relations Explorer

A Python-based e-reader and analytical QA engine for processing financial reports from Equinor's investor relations.

## Setup

1. Install dependencies:

```bash
uv sync
```

1. Install dependencies:

```bash
uv sync
```

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your-actual-api-key-here
```

1. (Optional) Customize model settings in `config.py`:

```python
# Change models for different quality/cost tradeoffs
MODEL_EXTRACTOR = "gpt-4o"  # or "gpt-4o-mini"
MODEL_QA = "gpt-4o"         # or "gpt-4o-mini"
```

## Local LLM with Ollama (Optional)

This project supports using local vision models via [Ollama](https://ollama.ai/) for high-quality table extraction.

1. Install Ollama: Download and install from [ollama.ai](https://ollama.ai/).
2. Pull the model:

```bash
ollama run llava-phi3
```

1. Configure: In `src/investor_relations_scraper/config.py`, ensure `PDF_EXTRACTION_METHOD` is set to `"ollama-vision"`.

> **Note**: Ensure the Ollama app is running (check your menu bar) or run `ollama serve` in a separate terminal before starting extraction.

## Usage

### Interactive Gradio UI (Recommended)

The interactive way to explore and extract data is via the Gradio E-Reader Frontend:

```bash
uv run python app.py
```

This provides a three-column layout where you can:

- Browse PDF pages alongside extracted text and tables.
- Interactively extract tables from specific pages using local vision models.
- **Edit and Save Tables**: Modifying and saving a table in the UI completely overwrites the existing table for that page and immediately syncs the changes to the underlying DuckDB database, providing instant updates for the QA chat.
- Ask questions to the agentic QA chat interface.
- Add and persist page-specific notes.

## Features

- **Interactive E-Reader**: Gradio-based web UI for reading, interactive extracting, and QA chat
- **Agentic QA**: Plan → Retrieve → Synthesize pipeline for answering financial questions with both text and embedded CSV tables
- **Error Handling**: Robust error handling
- **Progress Tracking**: Shows processing tracking and status

## Project Structure

```text
investor_relations_scraper/
├── app.py                         # Gradio interactive E-Reader UI
├── src/
│   └── investor_relations_scraper/
│       ├── config.py              # Central configuration
│       ├── cli.py                 # PDF extractor CLI
│       ├── document_loader.py     # Document loading & chunking
│       ├── search.py              # FAISS vector store & hybrid search
│       ├── conversation_memory.py # Chat history management
│       ├── qa_engine.py           # QA orchestrator (Plan → Retrieve → Synthesize)
│       └── extractors/            # PDF extraction strategies
│           ├── base.py            # Abstract base classes
│           ├── metadata_extractors.py
│           └── pdf_extractors.py  # PdfPlumber and Ollama local vision
├── examples/                      # Example usage scripts
├── tests/                         # Test suite
├── data/                          # Data directory
├── docs/                          # Documentation
├── pyproject.toml                 # Project dependencies
└── README.md                      # This file
```

## Requirements

- Python >= 3.9
- aiohttp
- OpenAI API key (for extractor)

All dependencies are managed via the `pyproject.toml` file.

## PDF Extractor

The extractor processes downloaded PDFs and uses OpenAI to clean and structure the data. Two extraction methods are available (set in `config.py`):

| Method | `PDF_EXTRACTION_METHOD` | Description |
| --- | --- | --- |
| **pdfplumber** | `"pdfplumber"` | Fast, text-based extraction |
| **Ollama Vision** | `"ollama-vision"` | Local OCR via `llava-phi3` (Requires Ollama) |

### Extractor Usage

By default, the extractor CLI processes both text and tables.

Process a single PDF:

```bash
uv run ir-extractor --file "Q1-2025-report.pdf"
```

Process all PDFs in `data/raw`:

```bash
uv run ir-extractor
```

Skip table extraction for faster processing:

```bash
uv run ir-extractor --skip-tables
```

### Extractor Output

The extractor creates:

- **Text files**: `data/processed/{filename}_text.txt`
- **Tables**: `data/processed/{filename}_table_{n}.csv`

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [**System Architecture**](docs/architecture.md): High-level design and module breakdown.
- [**Cheat Sheet**](docs/cheat_sheet.md): Quick reference for common commands.
