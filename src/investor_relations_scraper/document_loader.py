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
        
    def _generate_table_summary(self, csv_content: str, filename: str) -> str:
        """
        Generate a short LLM summary describing what a CSV table contains.
        
        Args:
            csv_content: The CSV content as a string
            filename: Name of the CSV file
            
        Returns:
            A 1-2 sentence summary of the table's contents
        """
        # Extract base info from filename
        base_name = re.sub(r'_table_(?:p\d+_)?\d+\.csv$', '', filename)
        
        prompt = f"""Analyze this financial table and describe what data it contains in 1-2 sentences.
Focus on: what metrics/figures are shown, what time period, and what category of data (e.g., revenue, expenses, dividends, production).

Filename: {filename}
Table content:
{csv_content[:2000]}

Description:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=config.MODEL_METADATA,
                messages=[
                    {"role": "system", "content": "You are a financial data analyst. Describe table contents concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠ Table summary generation failed for {filename}: {e}")
            # Fallback: use column headers
            first_line = csv_content.split('\n')[0] if csv_content else filename
            return f"Financial table from {base_name}: columns {first_line}"
    
    def _process_csv_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a single CSV table file into a document chunk with metadata.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List with a single document chunk, or empty list if table is too small
        """
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"  ⚠ Error reading CSV {file_path.name}: {e}")
            return []
        
        # Skip tables that are too small
        if df.shape[0] < config.MIN_TABLE_ROWS_FOR_INDEX:
            return []
        
        # Skip tables with no meaningful data (all NaN)
        if df.dropna(how='all').empty:
            return []
        
        # Convert to Markdown table for embedding
        markdown_table = df.to_markdown(index=False)
        
        # Generate LLM summary
        csv_content = file_path.read_text(encoding='utf-8')
        print(f"  Generating table summary...")
        table_summary = self._generate_table_summary(csv_content, file_path.name)
        print(f"  -> {table_summary}")
        
        # Extract metadata from filename (e.g., Q1-2025-report_table_5.csv)
        file_metadata = self._extract_metadata_from_filename(file_path.name)
        
        # Derive the original PDF source name
        source_name = re.sub(r'_table_(?:p\d+_)?\d+\.csv$', '.pdf', file_path.name)
        
        # Create a single chunk: summary + table content
        chunk_text = f"Table Summary: {table_summary}\n\n{markdown_table}"
        
        return [{
            "text": chunk_text,
            "metadata": {
                "source": source_name,
                "title": table_summary,
                "doc_type": "table",
                "quarter": file_metadata.get("quarter"),
                "year": file_metadata.get("year"),
                "company": file_metadata.get("company"),
                "table_summary": table_summary,
                "path": str(file_path),
                "chunk_id": 0
            }
        }]
    
    def extract_text_from_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        Load processed text files and CSV tables from directory
        
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
        
        # === Load CSV table files ===
        csv_files = list(path.glob('**/*_table_*.csv'))
        print(f"Found {len(csv_files)} CSV table files in {directory}")
        
        for file_path in csv_files:
            try:
                print(f"Loading table {file_path.name}...")
                chunks = self._process_csv_file(file_path)
                documents.extend(chunks)
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                
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
