"""
Abstract base classes for extraction strategies.

This module defines the interfaces that all metadata and PDF extractors must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


class BaseMetadataExtractor(ABC):
    """Abstract base class for metadata extraction strategies."""
    
    @abstractmethod
    def extract_metadata(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Extract metadata from document content and/or filename.
        
        Args:
            content: The text content of the document (may be partial)
            filename: The name of the file
            
        Returns:
            Dictionary containing metadata fields:
                - title: Document title
                - quarter: Quarter (Q1, Q2, Q3, Q4) or None
                - year: Year (e.g., "2024") or None
                - doc_type: Document type (report, transcript, etc.)
                - company: Company name or None
        """
        pass


class BasePDFExtractor(ABC):
    """Abstract base class for PDF extraction strategies."""
    
    @abstractmethod
    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        pass
    
    @abstractmethod
    def extract_tables(self, pdf_path: Path) -> List[Tuple[int, pd.DataFrame]]:
        """
        Extract tables from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of tuples (page_number, DataFrame) for each table found
        """
        pass
    
    @abstractmethod
    def supports_ocr(self) -> bool:
        """
        Indicate whether this extractor uses OCR capabilities.
        
        Returns:
            True if this extractor uses OCR, False otherwise
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this extractor for logging/display purposes.
        
        Returns:
            Human-readable name of the extractor
        """
        pass
