# Technical Documentation: Investor Relations Scraper & QA Engine

## 1. Solution Overview

This solution is an automated system designed to scrape, process, and analyze financial reports from Equinor's Investor Relations website. It creates a searchable knowledge base using Retrieval-Augmented Generation (RAG) to answer user queries about the company's financial performance.

## 2. System Architecture

The solution consists of three main modules:

1. **Scraper**: Automates the periodic retrieval of new financial documents.
2. **Extractor**: Converts raw PDFs into clean, machine-readable text and structured data.
3. **QA Engine**: Indexes the processed data and provides a conversational interface for querying.

### Directory Structure

```
investor_relations_scraper/
├── src/
│   └── investor_relations_scraper/
│       ├── config.py              # Central configuration
│       ├── scraper.py             # Playwright scraper
│       ├── cli.py                 # PDF extractor CLI
│       ├── document_loader.py     # Document loading, chunking & metadata
│       ├── search.py              # FAISS vector store & hybrid search engine
│       ├── conversation_memory.py # Chat history management
│       ├── qa_engine.py           # QA orchestrator (Plan → Retrieve → Synthesize)
│       └── extractors/            # PDF extraction strategies
│           ├── base.py            # Abstract base classes
│           ├── metadata_extractors.py
│           └── pdf_extractors.py  # PdfPlumber, Qwen2.5-VL, Fallback
├── data/
│   ├── raw/            # Downloaded PDFs
│   ├── processed/      # Cleaned text and CSV tables
│   ├── interim/        # Temporary files
│   └── vector_db/      # FAISS index and BM25 data
├── examples/           # Example usage scripts
├── tests/              # Test suite
└── docs/               # Documentation
```

## 3. Technologies & Implementation Details

### 3.1. Scraper Module (`scraper.py`)

* **Technology**: `Playwright` (Async API), `aiohttp`.
* **Target**: Equinor Quarterly Results Page.
* **Mechanism**:
  * Navigates to the Next.js-powered website using a headless browser.
  * Extracts PDF metadata directly from the `__NEXT_DATA__` JSON script tag to avoid parsing fragile DOM elements.
  * Uses `aiohttp` for reliable, high-speed direct file downloads.
  * Supports filtering by year (e.g., "2024", "2025").

### 3.2. Extractor Module (`cli.py` + `extractors/`)

* **Technology**: `pdfplumber`, `Qwen2.5-VL` (via `llama-cpp-python`), `OpenAI API` (`gpt-4o`).
* **Extraction Methods** (configured via `PDF_EXTRACTION_METHOD` in `config.py`):

| Method | Value | Description |
|--------|-------|-------------|
| **pdfplumber** | `"pdfplumber"` | Fast, text-based extraction for digital PDFs |
| **Qwen2.5-VL** | `"qwen-vl"` | Full OCR using a vision-language model for scanned PDFs |
| **Fallback** (default) | `"fallback"` | Uses pdfplumber first; for pages returning no text, falls back to Qwen2.5-VL OCR per-page. The Qwen model is lazily loaded only if needed. |

* **Process**:
    1. **Text Extraction**: Uses the configured extraction strategy (pdfplumber, Qwen OCR, or fallback).
    2. **Table Extraction**: Identifies and extracts tables using `pdfplumber`'s lattice/stream modes.
    3. **Cleaning & Structuring**:
        * Uses OpenAI's `gpt-4o` to clean raw text (fix formatting, remove artifacts).
        * Converts extracted tables into clean CSV format.
    4. **Output Generation**:
        * `{filename}_text.txt`: Clean, embedding-ready text.
        * `{filename}_table_{n}.csv`: Structured tabular data.

**Why the fallback approach?**

* **Fast path**: For fully digital PDFs, only pdfplumber runs — no model loading, no GPU usage.
* **Robust**: Scanned or image-based pages are automatically handled via OCR.
* **Lazy loading**: The Qwen vision model is only initialized if at least one page needs OCR.

### 3.3. QA Engine

The QA engine is split across four focused modules:

| Module | Class(es) | Responsibility |
|--------|-----------|----------------|
| `document_loader.py` | `ProcessedDocumentLoader` | Loads text/CSV files, chunks them, extracts metadata via LLM |
| `search.py` | `PersistentVectorStore`, `HybridSearchEngine` | FAISS index persistence, hybrid semantic + BM25 search |
| `conversation_memory.py` | `ConversationMemory` | Chat history with disk persistence |
| `qa_engine.py` | `QAEngine` | Orchestrates the Plan → Retrieve → Synthesize pipeline |

#### A. Metadata Extraction (`document_loader.py`)

* **Model**: `gpt-4o` (High capabilities).
* **Method**: Reads the first 3000 characters of a document to intelligently categorize it by:
  * **Quarter** (Q1, Q2, Q3, Q4)
  * **Year** (2024, 2025, etc.)
  * **Document Type** (Report, Transcript, Presentation, Financial Statements)
  * **Company Name** and **Title**.
* **Fallback**: Uses Regex on the filename if LLM extraction fails.

#### B. Indexing & Storage (`search.py`)

* **Vector Database**: `FAISS` (Facebook AI Similarity Search) using `IndexFlatIP` (Inner Product/Cosine Similarity).
* **Embeddings**: `BAAI/bge-small-en-v1.5` (via `sentence-transformers`). Optimized for MPS (Mac Metal) acceleration.
* **Keyword Index**: `BM25Okapi` (Rank-BM25) for traditional keyword matching.
* **Persistence**: Indices and metadata are pickled and saved to `data/vector_db/`.

#### C. Hybrid Search Strategy (`search.py`)

The engine uses a sophisticated two-stage search:

1. **Metadata Filtering (Stage 1)**: Scans all documents and assigns a boost score based on metadata matches (e.g., if query contains "Q1 2024", documents matching that quarter/year get a score boost).
2. **Hybrid Content Search (Stage 2)**:
    * **Semantic Search**: Uses vector embeddings to find conceptually similar text.
    * **Keyword Search**: Uses BM25 to find exact keyword matches.
    * **Combination**: `Score = (Alpha * Semantic_Score) + ((1-Alpha) * BM25_Score) + Metadata_Boost`.
    * *Default Alpha*: 0.7 (favors semantic search).

#### D. Agentic QA Pipeline (`qa_engine.py`)

The `QAEngine` uses a three-stage agentic pipeline:

1. **Plan**: Uses the LLM to analyze the document catalog and decide which sources are relevant to the question.
2. **Retrieve**: Fetches the most relevant chunks from the planned sources using hybrid search.
3. **Synthesize**: Generates a comprehensive answer with source citations.

* **Model**: `gpt-4o-mini`.
* **Memory**: Maintains a conversation history (persisted to disk) to allow follow-up questions.

## 4. Configuration (`config.py`)

The system is highly configurable via `config.py` and `.env` variables:

* **Models**: Switch between `gpt-4o` and `gpt-4o-mini` for different tasks to balance cost/performance.
* **PDF Extraction**: Choose between `pdfplumber`, `qwen-vl`, or `fallback` mode.
* **Processing**: Adjustable chunk sizes, overlap, and token limits.
* **Device**: Supports `cpu`, `cuda`, and `mps` (Mac).

## 5. Dependencies

* `openai`: LLM interactions.
* `playwright`: Web scraping.
* `pdfplumber`: PDF text and table extraction.
* `faiss-cpu` / `torch` / `sentence-transformers`: Vector search and embeddings.
* `rank-bm25`: Keyword search.
* `pandas` / `numpy`: Data manipulation.
* `llama-cpp-python` / `pdf2image` (optional): Qwen2.5-VL OCR fallback.
