# Financial Document Explorer

A Python-based e-reader and analytical QA engine for processing financial reports and investor relations documents.

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Copy the example env file and add your OpenAI API key:

```bash
cp .env.example .env
# Edit .env: OPENAI_API_KEY=your-actual-api-key-here
```

3. (Optional) Customize model settings in `config.py`:

```python
# Change models for different quality/cost tradeoffs
MODEL_EXTRACTOR = "gpt-4o"  # or "gpt-4o-mini"
MODEL_QA = "gpt-4o"         # or "gpt-4o-mini"
```

## Interactive Gradio UI (Recommended)

The interactive way to explore and extract data is via the Gradio E-Reader Frontend:

```bash
uv run python app.py
```

## Usage

This provides a three-column layout where you can:

- **Upload New Documents**: Use the "Upload PDF" button to add any financial report to the system. It will be automatically converted to text, cleaned by AI, and indexed for immediate QA.
- Browse PDF pages alongside extracted text and tables.
- Interactively extract tables from specific pages using local vision models.
- **Edit and Save Tables**: Modifying and saving a table in the UI completely overwrites the existing table for that page and immediately syncs the changes to the underlying DuckDB database, providing instant updates for the QA chat.
- Ask questions to the agentic QA chat interface.
- Add and persist page-specific notes.

## Features

- **Interactive E-Reader**: Gradio-based web UI for reading, interactive extracting, and QA chat.
- **Direct PDF Upload**: Upload and process new documents directly from the web interface.
- **Company-Aware Agentic QA**: Plan → Retrieve → Synthesize pipeline that understands multiple companies and cites specific sources.
- **Visual Table Extraction**: On-demand table extraction using OpenAI Vision models.
- **Progress Tracking**: Real-time status updates during document processing.

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

### PDF Extractor

The extractor processes downloaded PDFs and uses OpenAI to clean and structure the data.

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
