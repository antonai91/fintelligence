"""
Extraction strategies for PDF processing and metadata extraction.

This package provides abstract base classes and concrete implementations
for different extraction strategies.
"""

from .base import BaseMetadataExtractor, BasePDFExtractor
from .metadata_extractors import (
    LLMMetadataExtractor,
    RegexMetadataExtractor,
    HybridMetadataExtractor,
)
from .pdf_extractors import PdfPlumberExtractor, Qwen25VLExtractor, FallbackPDFExtractor

__all__ = [
    "BaseMetadataExtractor",
    "BasePDFExtractor",
    "LLMMetadataExtractor",
    "RegexMetadataExtractor",
    "HybridMetadataExtractor",
    "PdfPlumberExtractor",
    "Qwen25VLExtractor",
    "FallbackPDFExtractor",
]
