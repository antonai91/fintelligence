"""
Document Loader for Processed Text and CSV Files

Loads processed text files and CSV tables, chunks them with rich metadata
extracted via LLM or regex fallback.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from openai import OpenAI

from . import config


class ProcessedDocumentLoader:
    """Loads processed text files and chunks them with rich metadata"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._openai_client = None
        self._metadata_cache = {}  # Cache to avoid re-extracting for same files
        
    @property
    def openai_client(self):
        """Lazy initialization of OpenAI client for metadata extraction"""
        if self._openai_client is None:
            api_key = config.get_openai_api_key()
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client
        
    def _extract_metadata_with_llm(self, content: str, filename: str) -> Dict[str, str]:
        """
        Extract metadata from document content using LLM (GPT-4)
        
        This analyzes the actual content of the document to extract:
        - Quarter (Q1, Q2, Q3, Q4)
        - Year (2024, 2025, etc.)
        - Document type (report, transcript, presentation, financial-statements)
        - Company name
        - A descriptive title
        """
        # Use first ~3000 chars for metadata extraction (usually contains title, date, etc.)
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
            response = self.openai_client.chat.completions.create(
                model=config.MODEL_METADATA,
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
            # Fallback to filename-based extraction
            return self._extract_metadata_from_filename(filename)
    
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """
        Fallback: Extract metadata from filename using regex patterns
        
        Examples:
            Q1-2025-report_text.txt -> {quarter: Q1, year: 2025, doc_type: report}
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
        
        return metadata
        
    def extract_text_from_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        Load processed text files from directory
        
        Returns:
            List of documents (chunks) with rich metadata
        """
        documents = []
        path = Path(directory)
        
        # === Load text files ===
        text_files = list(path.glob('**/*_text.txt'))
        if not text_files:
            text_files = list(path.glob('**/*.txt'))
        
        print(f"Found {len(text_files)} processed text files in {directory}")
        
        for file_path in text_files:
            try:
                print(f"Loading {file_path.name}...")
                chunks = self._process_file(file_path)
                documents.extend(chunks)
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                
        # Note: CSV table files are now handled natively by the DuckDB SQL Agent
        # and are no longer chunked and embedded into the vector FAISS database.
                
        return documents
    
    def _process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single text file into chunks with metadata"""
        # Read the processed text
        text = file_path.read_text(encoding='utf-8')
        
        # Extract metadata using LLM (analyzes content, not just filename)
        print(f"  Extracting metadata with LLM...")
        file_metadata = self._extract_metadata_with_llm(text, file_path.name)
        print(f"  -> {file_metadata.get('title')} | {file_metadata.get('quarter')} {file_metadata.get('year')} | {file_metadata.get('doc_type')}")
        
        chunks = []
        
        # Simple word-based chunking
        words = text.split()
        
        chunk_id = 0
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Skip very small chunks
            if len(chunk_words) < 50:
                continue
            
            # Create chunk with rich metadata (including company from LLM)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": file_path.name.replace('_text.txt', '.pdf'),  # Original PDF name
                    "title": file_metadata["title"],
                    "doc_type": file_metadata["doc_type"],
                    "quarter": file_metadata["quarter"],
                    "year": file_metadata["year"],
                    "company": file_metadata.get("company"),
                    "path": str(file_path),
                    "chunk_id": chunk_id
                }
            })
            chunk_id += 1
            
        return chunks
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
