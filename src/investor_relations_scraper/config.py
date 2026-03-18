"""
Central Configuration File for Investor Relations Scraper

This file contains all configuration settings including:
- OpenAI model selections for different components
- API settings
- Directory paths
- Processing parameters
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set MPS (Metal) memory settings for Mac to prevent buffer errors
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ============================================================================
# API Configuration
# ============================================================================

# OpenAI API Key (loaded from .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================================
# Model Configuration
# ============================================================================

# Model for PDF text extraction and cleaning
# Options: "gpt-4o-mini" (faster, cheaper) or "gpt-4o" (higher quality)
MODEL_EXTRACTOR = "gpt-4o-mini"

# Model for Q&A engine
# Options: "gpt-4o-mini" (faster, cheaper) or "gpt-4o" (higher quality)
MODEL_QA = "gpt-4o-mini"

# Model for metadata extraction from document content
# Uses GPT-4 for high-quality metadata extraction
MODEL_METADATA = "gpt-4o-mini"

# Model for embeddings
# Using BAAI/bge-small-en-v1.5 (lightweight, fast, excellent for RAG)
MODEL_EMBEDDING = "BAAI/bge-small-en-v1.5"

# Device for local models ("cpu", "mps", "cuda", or None for auto)
MODEL_DEVICE = None

# ============================================================================
# Extraction Strategy Configuration
# ============================================================================

# Metadata extraction method
# Options: "llm" (intelligent, uses OpenAI), "regex" (fast, filename-based), "hybrid" (tries LLM, falls back to regex)
METADATA_EXTRACTION_METHOD = "hybrid"

# ============================================================================
# Vision Table Extraction Configuration (on-demand from UI)
# ============================================================================

# Model for vision-based table extraction from PDF page images via the OpenAI API
# Tables are only extracted on demand from the Gradio UI, never in batch.
MODEL_TABLE_EXTRACTOR = "gpt-4o-mini"

# Model for local vision-based table extraction via mlx-vlm (runs on Apple Silicon)
# Uses GLM-OCR converted to MLX format for zero-cost offline table extraction
MODEL_TABLE_EXTRACTOR_LOCAL = "EZCon/GLM-OCR-mlx"

# Maximum tokens for GLM-OCR table generation
TABLE_EXTRACTOR_MAX_TOKENS = 8192
# ============================================================================
# Directory Configuration
# ============================================================================

# Base project directory (go up two levels from src/investor_relations_scraper/)
BASE_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# DuckDB Database Path
TABLE_DB_PATH = VECTOR_DB_DIR / "tables.duckdb"

# ============================================================================
# Processing Configuration
# ============================================================================

# Maximum characters to process per PDF (to avoid token limits)
MAX_TEXT_CHARS = 100000

# Temperature for OpenAI API calls (0.0 = deterministic, 1.0 = creative)
TEMPERATURE = 0.1

# Batch size for processing multiple files
BATCH_SIZE = 10

# Batch size for generating embeddings
EMBEDDING_BATCH_SIZE = 32

# Maximum number of PDFs to process concurrently
MAX_CONCURRENT_PDFS = 10

# Maximum number of documents the agentic planner can select
MAX_PLANNED_SOURCES = 10

# Default number of chunks to retrieve per planned source
CHUNKS_PER_SOURCE_DEFAULT = 3

# Minimum number of data rows for a CSV table to be indexed
MIN_TABLE_ROWS_FOR_INDEX = 2

# ============================================================================
# Extractor Configuration
# ============================================================================

# Whether to skip text extraction by default
SKIP_TEXT_DEFAULT = False

# Whether to skip table extraction by default
SKIP_TABLES_DEFAULT = False

# ============================================================================
# Q&A Engine Configuration
# ============================================================================

# Number of documents to retrieve for context
TOP_K_DOCUMENTS = 5

# Maximum tokens for Q&A response
MAX_RESPONSE_TOKENS = 1000

# Document name boost for two-stage search (0-1)
# Higher values give more weight to documents matching the query metadata
DOC_NAME_BOOST = 0.3

# Processed text file pattern
PROCESSED_TEXT_PATTERN = "*_text.txt"

# ============================================================================
# Helper Functions
# ============================================================================

def get_openai_api_key() -> str:
    """
    Get OpenAI API key from environment
    
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not set
    """
    api_key = OPENAI_API_KEY
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file"
        )
    return api_key


def ensure_directories():
    """Create all necessary directories if they don't exist"""
    for directory in [RAW_DIR, PROCESSED_DIR, INTERIM_DIR, VECTOR_DB_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Configuration Summary
# ============================================================================

def print_config():
    """Print current configuration settings"""
    print("=" * 60)
    print("CONFIGURATION SETTINGS")
    print("=" * 60)
    print(f"Extractor Model: {MODEL_EXTRACTOR}")
    print(f"Q&A Model: {MODEL_QA}")
    print(f"Metadata Model: {MODEL_METADATA}")
    print(f"Table Extractor Model: {MODEL_TABLE_EXTRACTOR}")
    print(f"Embedding Model: {MODEL_EMBEDDING}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max Text Chars: {MAX_TEXT_CHARS:,}")
    print(f"Top K Documents: {TOP_K_DOCUMENTS}")
    print(f"Max Concurrent PDFs: {MAX_CONCURRENT_PDFS}")
    print(f"Raw Directory: {RAW_DIR}")
    print(f"Processed Directory: {PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    # Test configuration
    print_config()
    
    # Test API key
    try:
        api_key = get_openai_api_key()
        print(f"\n✓ OpenAI API Key: {api_key[:20]}...")
    except ValueError as e:
        print(f"\n✗ Error: {e}")
