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
)

__all__ = [
    "BaseMetadataExtractor",
    "BasePDFExtractor",
    "LLMMetadataExtractor",
    "RegexMetadataExtractor",
    "HybridMetadataExtractor",
    "PdfPlumberExtractor",
    "Qwen25VLExtractor",
]
