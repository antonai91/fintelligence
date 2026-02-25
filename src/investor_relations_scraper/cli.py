"""
PDF Extractor for Investor Relations Reports

This script extracts text and tables from PDF files using pdfplumber,
then processes them with OpenAI API to create clean, embedding-ready text
and well-formatted CSV files.
"""

import argparse
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncio
import pandas as pd
import pdfplumber
from openai import AsyncOpenAI

from . import config
from .extractors import PdfPlumberExtractor

# Constants
MIN_TABLE_ROWS = 2
TABLE_INDEX_START = 1
PAGE_NUMBER_START = 1


class PDFExtractor:
    """Extract and process text and tables from PDF files"""
    
    def __init__(
        self,
        raw_dir: Optional[str] = None,
        processed_dir: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the PDF extractor
        
        Args:
            raw_dir: Directory containing raw PDF files (defaults to config.RAW_DIR)
            processed_dir: Directory to save processed output (defaults to config.PROCESSED_DIR)
            model: OpenAI model to use (defaults to config.MODEL_EXTRACTOR)
            api_key: OpenAI API key (defaults to config.OPENAI_API_KEY)
        """
        self.raw_dir = Path(raw_dir) if raw_dir else config.RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else config.PROCESSED_DIR
        self.model = model or config.MODEL_EXTRACTOR
        
        # Initialize OpenAI client
        try:
            api_key = api_key or config.get_openai_api_key()
            self.client = AsyncOpenAI(api_key=api_key)
        except ValueError as e:
            print(f"❌ Error: {e}")
            raise

        self.extractor = PdfPlumberExtractor()
        print(f"   Extraction Strategy: {self.extractor.get_name()}")

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        max_pdfs = getattr(config, "MAX_CONCURRENT_PDFS", 10)
        self.semaphore = asyncio.Semaphore(max_pdfs)
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract raw text from PDF file using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string, or empty string if extraction fails
        """
        return self.extractor.extract_text(pdf_path)
    
    async def clean_text_with_openai(self, text: str, filename: str) -> str:
        """
        Clean and process text using OpenAI API.

        Args:
            text: Raw extracted text
            filename: Name of the source file (for context)

        Returns:
            Cleaned text ready for embedding, or original text if cleaning fails
        """
        if not text.strip():
            return ""
        
        # Check if text needs chunking
        max_chars = config.MAX_TEXT_CHARS
        if len(text) > max_chars:
            print(f"  ℹ Text is long ({len(text)} chars), chunking into segments of ~{max_chars} chars")
            chunks = self._chunk_text(text, max_chars)
            print(f"  ℹ Split into {len(chunks)} chunks for processing")
            
            # Process chunks in parallel
            async def process_chunk(chunk_idx, chunk_text):
                # Add context about chunk position
                chunk_prompt_suffix = f" (Part {chunk_idx+1}/{len(chunks)})"
                return await self._clean_single_chunk(chunk_text, filename + chunk_prompt_suffix)

            tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
            cleaned_chunks = await asyncio.gather(*tasks)
            
            # Join chunks with double newline
            return "\n\n".join(cleaned_chunks)
        else:
            return await self._clean_single_chunk(text, filename)

    async def _clean_single_chunk(self, text: str, filename: str) -> str:
        """Process a single chunk of text with OpenAI."""
        prompt = self._build_text_cleaning_prompt(filename, text)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial document processing assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE,
            )
            
            cleaned_text = response.choices[0].message.content.strip()
            return cleaned_text
            
        except Exception as e:
            print(f"  ✗ Error cleaning text chunk with OpenAI: {e}")
            print("  ⚠ Falling back to raw text for this chunk")
            return text

    def _chunk_text(self, text: str, max_chars: int) -> List[str]:
        """
        Split text into chunks that respect paragraph boundaries.
        
        Args:
            text: Text to split
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split by double newlines (paragraphs) first
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            # If a single paragraph is too long, split by single newlines
            if len(paragraph) > max_chars:
                lines = paragraph.split('\n')
                for line in lines:
                    if current_length + len(line) + 1 > max_chars:
                        if current_chunk:
                            chunks.append('\n\n'.join(current_chunk))
                            current_chunk = []
                            current_length = 0
                    
                    # If a single line is still too long, we have to hard split it
                    if len(line) > max_chars:
                        # Process what we have so far
                        if current_chunk:
                            chunks.append('\n\n'.join(current_chunk))
                            current_chunk = []
                            current_length = 0
                        
                        # Hard split the long line
                        for i in range(0, len(line), max_chars):
                            chunks.append(line[i:i + max_chars])
                    else:
                        current_chunk.append(line)
                        current_length += len(line) + 1
            else:
                if current_length + len(paragraph) + 2 > max_chars:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(paragraph)
                current_length += len(paragraph) + 2
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks
    
    def _build_text_cleaning_prompt(self, filename: str, text: str) -> str:
        """Build the prompt for text cleaning."""
        return f"""You are processing a financial report PDF ({filename}) for text embedding and semantic search.

Your task:
1. Remove headers, footers, page numbers, and navigation elements
2. Remove formatting artifacts and excessive whitespace
3. Preserve all meaningful financial data, metrics, and narrative content
4. Maintain logical structure and paragraph breaks
5. Keep section headings and important labels
6. Preserve numbers, dates, and financial figures exactly as they appear

Return ONLY the cleaned text, without any explanations or metadata.

Raw text:
{text}"""
    
    async def process_table_with_openai(self, df: pd.DataFrame, page_num: int, table_num: int) -> Optional[pd.DataFrame]:
        """
        Clean and format table using OpenAI API.
        
        Args:
            df: DataFrame containing table data
            page_num: Page number where table was found
            table_num: Table number on the page
            
        Returns:
            Cleaned DataFrame, or original DataFrame if processing fails
        """
        try:
            table_str = df.to_string(index=False, na_rep='')
            prompt = self._build_table_cleaning_prompt(page_num, table_str)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial data processing assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE,
            )
            
            csv_content = response.choices[0].message.content.strip()
            csv_content = self._clean_markdown_code_blocks(csv_content)
            
            # Parse CSV back into DataFrame
            cleaned_df = pd.read_csv(StringIO(csv_content))
            return cleaned_df
            
        except Exception as e:
            print(f"  ✗ Error processing table with OpenAI: {e}")
            print("  ⚠ Falling back to raw table data")
            return df
    
    def _build_table_cleaning_prompt(self, page_num: int, table_str: str) -> str:
        """Build the prompt for table cleaning."""
        return f"""You are processing a table from page {page_num} of a financial report PDF.

Your task:
1. Clean the table data by removing artifacts and formatting issues
2. Identify proper column headers (usually in the first row)
3. Ensure data is properly aligned with headers
4. Handle merged cells by filling in appropriate values
5. Remove completely empty rows or columns
6. Preserve all numerical data exactly as shown
7. Return the table in CSV format with proper headers

Original table:
{table_str}

Return ONLY the CSV data (with headers), nothing else."""
    
    def _clean_markdown_code_blocks(self, content: str) -> str:
        """Remove markdown code block markers from content."""
        content = re.sub(r'^```csv\n', '', content)
        content = re.sub(r'^```\n', '', content)
        content = re.sub(r'\n```$', '', content)
        return content
    
    async def process_pdf(
        self,
        pdf_path: Path,
        skip_text: bool = False,
        clean: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single PDF file (text only; tables are extracted on demand from the UI).
        
        Args:
            pdf_path: Path to PDF file
            skip_text: Skip text extraction
            clean: Clean text with OpenAI
            
        Returns:
            Dictionary with processing results including files created and errors
        """
        print(f"\n📄 Processing: {pdf_path.name}")
        
        base_name = pdf_path.stem
        results = {
            "pdf": pdf_path.name,
            "text_file": None,
            "errors": []
        }
        
        if not skip_text:
            await self._process_text(pdf_path, base_name, results, clean)
        
        return results
    
    async def _process_text(self, pdf_path: Path, base_name: str, results: Dict[str, Any], clean: bool = True) -> None:
        """Extract and process text from PDF."""
        print("  📝 Extracting text...")
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        if not raw_text:
            results["errors"].append("No text extracted")
            return
            
        if clean:
            print(f"  🤖 Cleaning text with OpenAI ({self.model})...")
            final_text = await self.clean_text_with_openai(raw_text, pdf_path.name)
        else:
            print("  ⏩ Skipping text cleaning (saving raw text)")
            final_text = raw_text

        if final_text:
            text_file = self.processed_dir / f"{base_name}_text.txt"
            text_file.write_text(final_text, encoding='utf-8')
            results["text_file"] = text_file.name
            print(f"  ✓ Saved text: {text_file.name}")
        else:
            results["errors"].append("Failed to process text")
    

    async def process_all_pdfs(
        self,
        skip_text: bool = False,
        clean: bool = True,
        file_pattern: str = "*.pdf"
    ) -> List[Dict[str, Any]]:
        """
        Process all PDF files in the raw directory (text only).

        Args:
            skip_text: Skip text extraction
            clean: Whether to clean text with OpenAI
            file_pattern: Glob pattern for PDF files (default: "*.pdf")

        Returns:
            List of processing results for each file
        """
        pdf_files = list(self.raw_dir.glob(file_pattern))

        if not pdf_files:
            print(f"⚠ No PDF files found in {self.raw_dir}")
            return []

        print(f"Found {len(pdf_files)} PDF files to process")

        async def bounded_process_pdf(idx, pdf_path):
            async with self.semaphore:
                print(f"\n[{idx}/{len(pdf_files)}] Starting {pdf_path.name}")
                return await self.process_pdf(pdf_path, skip_text, clean)

        tasks = [bounded_process_pdf(idx, pdf_path) for idx, pdf_path in enumerate(pdf_files, TABLE_INDEX_START)]
        all_results = await asyncio.gather(*tasks)

        return all_results

    
    def print_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print processing summary."""
        separator = "=" * 60
        print(f"\n{separator}")
        print("PROCESSING SUMMARY")
        print(separator)
        
        total_files = len(results)
        text_files = sum(1 for r in results if r["text_file"])
        errors = sum(len(r["errors"]) for r in results)
        
        print(f"Total PDFs processed: {total_files}")
        print(f"Text files created: {text_files}")
        print(f"Errors encountered: {errors}")
        print(f"\nOutput directory: {self.processed_dir.absolute()}")
        
        if errors > 0:
            print("\nFiles with errors:")
            for r in results:
                if r["errors"]:
                    print(f"  • {r['pdf']}: {', '.join(r['errors'])}")


async def main_async() -> None:
    """Main async entry point."""
    parser = argparse.ArgumentParser(
        description='Extract text and tables from PDF files using OpenAI API'
    )
    parser.add_argument(
        '--raw-dir',
        help=f'Directory containing raw PDF files (default: {config.RAW_DIR})'
    )
    parser.add_argument(
        '--processed-dir',
        help=f'Directory to save processed files (default: {config.PROCESSED_DIR})'
    )
    parser.add_argument(
        '--model',
        choices=['gpt-4o-mini', 'gpt-4o'],
        help=f'OpenAI model to use (default: {config.MODEL_EXTRACTOR})'
    )
    parser.add_argument(
        '--file',
        help='Process only this specific file (e.g., Q1-2025-report.pdf)'
    )
    parser.add_argument(
        '--skip-text',
        action='store_true',
        help='Skip text extraction'
    )
    parser.add_argument(
        '--no-cleaning',
        action='store_true',
        help='Skip OpenAI cleaning (faster, raw output)'
    )
    parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    
    args = parser.parse_args()
    clean = not args.no_cleaning
    
    # Print configuration
    print("\n📋 Current Configuration:")
    print(f"   Model: {args.model or config.MODEL_EXTRACTOR}")
    print(f"   Raw Dir: {args.raw_dir or config.RAW_DIR}")
    print(f"   Processed Dir: {args.processed_dir or config.PROCESSED_DIR}")
    print(f"   Cleaning Enabled: {clean}")
    print()
    
    # Initialize extractor (will validate API key)
    try:
        extractor = PDFExtractor(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            model=args.model,
            api_key=args.api_key
        )
    except ValueError:
        return
    
    # Process files
    if args.file:
        # Process single file
        pdf_path = extractor.raw_dir / args.file
        if not pdf_path.exists():
            print(f"❌ Error: File not found: {pdf_path}")
            return
        results = [await extractor.process_pdf(pdf_path, args.skip_text, clean)]
    else:
        # Process all files
        results = await extractor.process_all_pdfs(args.skip_text, clean)
    
    # Print summary
    extractor.print_summary(results)
    print("\n✓ Processing complete!")


def main():
    """Entry point wrapper."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n⚠ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ detailed error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
