"""
Concrete implementations of metadata extraction strategies.

This module provides different strategies for extracting metadata from documents:
- LLMMetadataExtractor: Uses an LLM to intelligently extract metadata
- RegexMetadataExtractor: Uses regex patterns on filenames (fast fallback)
- HybridMetadataExtractor: Tries LLM first, falls back to regex on failure
"""

import re
import json
from typing import Dict, Any, Optional
from openai import OpenAI

from .base import BaseMetadataExtractor


class LLMMetadataExtractor(BaseMetadataExtractor):
    """Extract metadata using an LLM to analyze document content."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize the LLM metadata extractor.
        
        Args:
            model: OpenAI model to use for extraction
            api_key: OpenAI API key (if None, will use environment variable)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
    
    def extract_metadata(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Extract metadata using LLM analysis of content and filename.
        
        Args:
            content: Document content (first ~3000 chars typically)
            filename: Name of the file
            
        Returns:
            Dictionary with metadata fields
        """
        # Use first ~3000 chars for metadata extraction
        sample_content = content[:3000] if len(content) > 3000 else content
        
        # Remove _text.txt suffix for base filename
        base_name = filename.replace('_text.txt', '').replace('.txt', '')
        
        prompt = f"""Analyze this financial document and extract the following metadata.
Return ONLY a JSON object with these fields (no markdown, no explanation):

{{
    "quarter": "Q1" or "Q2" or "Q3" or "Q4" or null (if not quarterly),
    "year": "2024" (4-digit year string) or null,
    "doc_type": one of ["report", "transcript", "presentation", "financial-statements", "annual-report", "other"],
    "company": "Company Name" or null,
    "title": "A short descriptive title for this document"
}}

Filename: {base_name}

Document content (first portion):
{sample_content}

JSON:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a metadata extraction assistant. Extract structured metadata from financial documents. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean up any markdown formatting
            if result.startswith("```"):
                result = re.sub(r'^```(?:json)?\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
            
            metadata = json.loads(result)
            
            # Ensure all expected fields exist
            return {
                "title": metadata.get("title") or base_name,
                "quarter": metadata.get("quarter"),
                "year": metadata.get("year"),
                "doc_type": metadata.get("doc_type") or "unknown",
                "company": metadata.get("company")
            }
            
        except Exception as e:
            print(f"  Warning: LLM metadata extraction failed for {filename}: {e}")
            # Return basic metadata from filename
            return {
                "title": base_name,
                "quarter": None,
                "year": None,
                "doc_type": "unknown",
                "company": None
            }


class RegexMetadataExtractor(BaseMetadataExtractor):
    """Extract metadata using regex patterns on filenames (fast fallback)."""
    
    def extract_metadata(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Extract metadata using regex patterns on filename.
        
        Args:
            content: Document content (not used by this extractor)
            filename: Name of the file
            
        Returns:
            Dictionary with metadata fields
        """
        # Remove _text.txt suffix
        base_name = filename.replace('_text.txt', '').replace('.txt', '')
        
        metadata = {
            "title": base_name,
            "quarter": None,
            "year": None,
            "doc_type": "unknown",
            "company": None
        }
        
        # Extract quarter (Q1, Q2, Q3, Q4)
        quarter_match = re.search(r'Q([1-4])', base_name, re.IGNORECASE)
        if quarter_match:
            metadata["quarter"] = f"Q{quarter_match.group(1)}"
        
        # Extract year (2024, 2025, etc.)
        year_match = re.search(r'(20\d{2})', base_name)
        if year_match:
            metadata["year"] = year_match.group(1)
        
        # Determine document type
        lower_name = base_name.lower()
        if 'transcript' in lower_name:
            metadata["doc_type"] = "transcript"
        elif 'presentation' in lower_name:
            metadata["doc_type"] = "presentation"
        elif 'report' in lower_name:
            metadata["doc_type"] = "report"
        elif 'financial' in lower_name or 'statement' in lower_name:
            metadata["doc_type"] = "financial-statements"
        elif 'annual' in lower_name:
            metadata["doc_type"] = "annual-report"
        
        return metadata


class HybridMetadataExtractor(BaseMetadataExtractor):
    """Try LLM extraction first, fall back to regex on failure."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize the hybrid metadata extractor.
        
        Args:
            model: OpenAI model to use for LLM extraction
            api_key: OpenAI API key (if None, will use environment variable)
        """
        self.llm_extractor = LLMMetadataExtractor(model=model, api_key=api_key)
        self.regex_extractor = RegexMetadataExtractor()
    
    def extract_metadata(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Extract metadata using LLM, fall back to regex if LLM fails.
        
        Args:
            content: Document content
            filename: Name of the file
            
        Returns:
            Dictionary with metadata fields
        """
        try:
            # Try LLM extraction first
            metadata = self.llm_extractor.extract_metadata(content, filename)
            
            # If LLM extraction returned minimal data, enhance with regex
            if metadata.get("quarter") is None or metadata.get("year") is None:
                regex_metadata = self.regex_extractor.extract_metadata(content, filename)
                
                # Fill in missing fields from regex
                if metadata.get("quarter") is None:
                    metadata["quarter"] = regex_metadata.get("quarter")
                if metadata.get("year") is None:
                    metadata["year"] = regex_metadata.get("year")
                if metadata.get("doc_type") == "unknown":
                    metadata["doc_type"] = regex_metadata.get("doc_type")
            
            return metadata
            
        except Exception as e:
            print(f"  Warning: LLM extraction failed, using regex fallback: {e}")
            return self.regex_extractor.extract_metadata(content, filename)
