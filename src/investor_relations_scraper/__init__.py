"""
Investor Relations Scraper Package

A Python package for scraping, processing, and analyzing financial reports
from investor relations websites.
"""

__version__ = "0.1.0"

from .config import *
from .extractors import (
    BaseMetadataExtractor,
    BasePDFExtractor,
    LLMMetadataExtractor,
    RegexMetadataExtractor,
    HybridMetadataExtractor,
    PdfPlumberExtractor,
    Qwen25VLExtractor,
    FallbackPDFExtractor,
)
from .scraper import EquinorScraper
from .qa_engine import QAEngine
from .document_loader import ProcessedDocumentLoader
from .search import PersistentVectorStore, HybridSearchEngine
from .conversation_memory import ConversationMemory
from .cli import PDFExtractor

__all__ = [
    "BaseMetadataExtractor",
    "BasePDFExtractor",
    "LLMMetadataExtractor",
    "RegexMetadataExtractor",
    "HybridMetadataExtractor",
    "PdfPlumberExtractor",
    "Qwen25VLExtractor",
    "FallbackPDFExtractor",
    "EquinorScraper",
    "QAEngine",
    "ProcessedDocumentLoader",
    "PersistentVectorStore",
    "HybridSearchEngine",
    "ConversationMemory",
    "PDFExtractor",
]
