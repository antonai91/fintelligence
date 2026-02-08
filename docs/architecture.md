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
├── config.py           # Central configuration
├── data/               # Data storage
│   ├── raw/            # Downloaded PDFs
│   ├── processed/      # Cleaned text and CSV tables
│   ├── interim/        # Temporary files
│   └── vector_db/      # FAISS index and BM25 data
└── scripts/
    ├── scraper.py      # Playwright scraper
    ├── extractor.py    # PDF text/table processor
    └── qa_engine.py    # RAG engine (Vector DB + Search)
```

## 3. Technologies & Implementation Details

### 3.1. Scraper Module (`scripts/scraper.py`)

* **Technology**: `Playwright` (Async API), `aiohttp`.
* **Target**: Equinor Quarterly Results Page.
* **Mechanism**:
  * Navigates to the Next.js-powered website using a headless browser.
  * Extracts PDF metadata directly from the `__NEXT_DATA__` JSON script tag to avoid parsing fragile DOM elements.
  * Uses `aiohttp` for reliable, high-speed direct file downloads.
  * supports filtering by year (e.g., "2024", "2025").

### 3.2. Extractor Module (`scripts/extractor.py`)

* **Technology**: `pdfplumber`, `OpenAI API` (`gpt-4o`).
* **Process**:
    1. **Text Extraction**: Iterates through PDF pages using `pdfplumber` to extract raw text.
    2. **Table Extraction**: Identifies and extracts tables using `pdfplumber`'s lattice/stream modes.
    3. **Cleaning & Structuring**:
        * Uses OpenAI's `gpt-4o` to clean raw text (fix formatting, remove artifacts).
        * Converts extracted tables into clean CSV format.
    4. **Output Generation**:
        * `{filename}_text.txt`: Clean, embedding-ready text.
        * `{filename}_table_{n}.csv`: Structured tabular data.

**Why this approach?**

* **Reliability**: Proven, stable library for text extraction.
* **Simplicity**: No need for large model downloads or complex GPU setup.
* **AI Enhancement**: LLM post-processing compensates for raw extraction limitations.

### 3.3. QA Engine (`scripts/qa_engine.py`)

This is the core RAG system.

#### A. Metadata Extraction

* **Model**: `gpt-4o` (High capabilities).
* **Method**: Reads the first 3000 characters of a document to intelligently categorize it by:
  * **Quarter** (Q1, Q2, Q3, Q4)
  * **Year** (2024, 2025, etc.)
  * **Document Type** (Report, Transcript, Presentation, Financial Statements)
  * **Company Name** and **Title**.
* **Fallback**: Uses Regex on the filename if LLM extraction fails.

#### B. Indexing & Storage

* **Vector Database**: `FAISS` (Facebook AI Similarity Search) using `IndexFlatIP` (Inner Product/Cosine Similarity).
* **Embeddings**: `BAAI/bge-small-en-v1.5` (via `sentence-transformers`). Optimized for MPS (Mac Metal) acceleration.
* **Keyword Index**: `BM25Okapi` (Rank-BM25) for traditional keyword matching.
* **Persistence**: Indices and metadata are pickled and saved to `data/vector_db/`.

#### C. Hybrid Search Strategy

The engine uses a sophisticated two-stage search:

1. **Metadata Filtering (Stage 1)**: Scans all documents and assigns a boost score based on metadata matches (e.g., if query contains "Q1 2024", documents matching that quarter/year get a score boost).
2. **Hybrid Content Search (Stage 2)**:
    * **Semantic Search**: Uses vector embeddings to find conceptually similar text.
    * **Keyword Search**: Uses BM25 to find exact keyword matches.
    * **Combination**: `Score = (Alpha * Semantic_Score) + ((1-Alpha) * BM25_Score) + Metadata_Boost`.
    * *Default Alpha*: 0.7 (favors semantic search).

#### D. Generation

* **Model**: `gpt-4o-mini`.
* **Context Window**: Retrieval of Top-5 most relevant chunks.
* **Memory**: Maintains a conversation history (persisted to disk) to allow follow-up questions.

## 4. Configuration (`config.py`)

The system is highly configurable via `config.py` and `.env` variables:

* **Models**: Switch between `gpt-4o` and `gpt-4o-mini` for different tasks to balance cost/performance.
* **Processing**: Adjustable chunk sizes, overlap, and token limits.
* **Device**: Supports `cpu`, `cuda`, and `mps` (Mac).

## 5. Dependencies

* `openai`: LLM interactions.
* `playwright`: Web scraping.
* `pdfplumber`: PDF text and table extraction.
* `faiss-cpu` / `torch` / `sentence-transformers`: Vector search and embeddings.
* `rank-bm25`: Keyword search.
* `pandas` / `numpy`: Data manipulation.
