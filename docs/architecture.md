# Technical Documentation: Financial Document Explorer & QA Engine

## 1. Solution Overview

This solution is an automated system designed to process and analyze financial reports and investor relations documents. It creates a searchable knowledge base using Retrieval-Augmented Generation (RAG) to answer user queries across multiple companies.

## 2. System Architecture

The solution consists of four main modules:

1. **Extractor**: Converts raw PDFs into clean, machine-readable text and structured data.
2. **Interactive UI**: A unified Gradio frontend for exploring, extracting tables, and chatting.
3. **QA Engine**: Indexes the processed data and provides a conversational interface for querying.

### Directory Structure

```text
investor_relations_scraper/
├── app.py                 # Gradio Interactive E-Reader Frontend
├── src/
│   └── investor_relations_scraper/
│       ├── config.py              # Central configuration
│       ├── cli.py                 # PDF extractor CLI
│       ├── document_loader.py     # Document loading, chunking & metadata
│       ├── search.py              # FAISS vector store & hybrid search engine
│       ├── conversation_memory.py # Chat history management
│       ├── qa_engine.py           # QA orchestrator (Plan → Retrieve → Synthesize)
│       ├── table_db.py            # DuckDB manager for table ingestion & Text-to-SQL
│       └── extractors/            # PDF extraction strategies
│           ├── base.py            # Abstract base classes
│           ├── metadata_extractors.py
│           └── pdf_extractors.py  # PdfPlumber and GPT-4o Vision
├── data/
│   ├── raw/            # Downloaded PDFs
│   ├── processed/      # Cleaned text and CSV tables
│   ├── interim/        # Temporary files
│   └── vector_db/      # FAISS index, BM25 data, and DuckDB tables (tables.duckdb)
├── examples/           # Example usage scripts
├── tests/              # Test suite
└── docs/               # Documentation
```

## 3. Technologies & Implementation Details

### 3.1. Extractor Module (`cli.py` + `extractors/`)

* **Technology**: `pdfplumber`, `OpenAI API` (`gpt-4o`/`gpt-4o-mini`).
* **Extraction Methods**:
  * **pdfplumber** (`PdfPlumberExtractor`): Fast, text-based extraction for digital PDFs; used for both CLI batch text and table structure detection.
  * **GPT-4o Vision** (`GPT4VisionExtractor`): Vision-based table extraction via OpenAI; used on-demand from the UI (and optionally for batch). Uses a pdfplumber pre-filter to send only pages with structural tables to the API.

* **Process**:
    1. **Text Extraction**: Uses pdfplumber for all batch text extraction.
    2. **Table Extraction**: CLI batch runs text-only by default; tables are extracted on demand in the UI (or via vision when configured). Table CSVs are synced into DuckDB for QA.
    3. **Cleaning & Structuring**:
        * Uses OpenAI's models to clean raw text (fix formatting, remove artifacts).
        * Converts extracted tables into clean CSV format.
    4. **Output Generation**:
        * `{filename}_text.txt`: Clean, embedding-ready text.
        * `{filename}_table_p{page}_{n}.csv`: Structured tabular data mapped to specific pages.

### 3.2. Interactive E-Reader UI (`app.py`)

* **Technology**: `Gradio`.
* **Purpose**: A unified explorer providing three-column synced capability:
  * Left side for viewing original rendered PDF pages, **Uploading new documents**, and Interactive Vision Table Extraction.
  * Right side for maintaining the AI Assistant chat interface.
* **Upload Pipeline**: The UI features an "Upload PDF" button that triggers an asynchronous pipeline:
    1. **Saves** the file to the local `data/raw/` directory.
    2. **Processes** the document using `PDFExtractor` to extract and clean text using GPT-4o-mini.
    3. **Indexes** the new content immediately into the Vector Store and DuckDB database.
    4. **Refreshes** the document selection for instant exploration.

### 3.3. QA Engine

The QA engine is split across four focused modules:

| Module | Class(es) | Responsibility |
| --- | --- | --- |
| `document_loader.py` | `ProcessedDocumentLoader` | Loads text/CSV files, chunks them, extracts metadata via LLM |
| `search.py` | `PersistentVectorStore`, `HybridSearchEngine` | FAISS index persistence, hybrid semantic + BM25 search |
| `conversation_memory.py` | `ConversationMemory` | Chat history with disk persistence |
| `qa_engine.py` | `QAEngine` | Orchestrates the Plan → Retrieve → Synthesize pipeline |

#### A. Metadata Extraction (`document_loader.py`)

* **Model**: `gpt-4o` (High capabilities).
* **Method**: Reads the first several thousand characters of a document to intelligently categorize it by:
  * **Quarter** (Q1, Q2, Q3, Q4)
  * **Year** (2024, 2025, etc.)
  * **Document Type** (Report, Transcript, Presentation, Financial Statements)
  * **Company Name** (Crucial for multi-company support)
  * **Title**.
* **Integration**: The `company` metadata is used both in the planning phase (deciding which source to use) and the synthesis phase (ensuring the AI knows whose data it is citing).
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

### E. Table Storage & Text-to-SQL (`table_db.py`)

* **Database**: DuckDB stores extracted table CSVs in `data/vector_db/tables.duckdb` (config: `TABLE_DB_PATH`).
* **Catalog**: A metadata table `_table_catalog` tracks CSV filename, SQL table name, source PDF, and schema for each ingested table.
* **Sync**: The Gradio UI and pipeline sync tables from `data/processed/*_table_*.csv` into DuckDB; edits saved in the UI overwrite the CSV and reload the corresponding DuckDB table.
* **QA Integration**: The QA engine can run Text-to-SQL over DuckDB when answering analytical questions across companies and tables.

## 4. Configuration (`config.py`)

The system is highly configurable via `config.py` and `.env` variables:

* **Models**: Switch between `gpt-4o` and `gpt-4o-mini` for extractor, QA, metadata, and table extraction (`MODEL_EXTRACTOR`, `MODEL_QA`, `MODEL_METADATA`, `MODEL_TABLE_EXTRACTOR`).
* **PDF Extraction**: Text extraction uses pdfplumber; vision-based table extraction uses OpenAI (`MODEL_TABLE_EXTRACTOR`, on-demand from UI).
* **Processing**: Adjustable chunk sizes, token limits, and DuckDB table indexing (`TABLE_DB_PATH`, `MIN_TABLE_ROWS_FOR_INDEX`).
* **Device**: Embedding model supports `cpu`, `cuda`, and `mps` (Mac) via `MODEL_DEVICE`.

## 5. Dependencies

* `openai`: LLM interactions.
* `pdfplumber`: PDF text and table extraction.
* `faiss-cpu` / `torch` / `sentence-transformers`: Vector search and embeddings.
* `rank-bm25`: Keyword search.
* `pandas` / `numpy`: Data manipulation.
* `gradio`: Web interface.
* `pdf2image` (optional): Required for rendering PDF pages as images for Ollama vision.
