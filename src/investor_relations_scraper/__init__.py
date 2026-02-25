"""
Investor Relations Package

A Python package for processing and analyzing financial reports.
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
    GPT4VisionExtractor,
)
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
    "GPT4VisionExtractor",
    "QAEngine",
    "ProcessedDocumentLoader",
    "PersistentVectorStore",
    "HybridSearchEngine",
    "ConversationMemory",
    "PDFExtractor",
]
