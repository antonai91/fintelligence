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
        
        # Create processed directory if it doesn't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent PDF processing
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract raw text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string, or empty string if extraction fails
        """
        text_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, PAGE_NUMBER_START):
                    text = page.extract_text()
                    if text:
                        text_content.append(f"--- Page {page_num} ---\n{text}")
                        
            return "\n\n".join(text_content)
        except (IOError, OSError) as e:
            print(f"✗ Error reading PDF file {pdf_path.name}: {e}")
            return ""
        except Exception as e:
            print(f"✗ Unexpected error extracting text from {pdf_path.name}: {e}")
            return ""
    
    def extract_tables_from_pdf(self, pdf_path: Path) -> List[Tuple[int, pd.DataFrame]]:
        """
        Extract tables from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of tuples (page_number, DataFrame), or empty list if extraction fails
        """
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, PAGE_NUMBER_START):
                    page_tables = page.extract_tables()
                    
                    for table_data in page_tables:
                        if not table_data:
                            continue
                            
                        df = pd.DataFrame(table_data)
                        
                        # Skip empty or single-row tables
                        if not df.empty and df.shape[0] >= MIN_TABLE_ROWS:
                            tables.append((page_num, df))
                                
            return tables
        except (IOError, OSError) as e:
            print(f"✗ Error reading PDF file {pdf_path.name}: {e}")
            return []
        except Exception as e:
            print(f"✗ Unexpected error extracting tables from {pdf_path.name}: {e}")
            return []
    
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
        
        # Truncate if text is too long (to avoid token limits)
        max_chars = config.MAX_TEXT_CHARS
        if len(text) > max_chars:
            print(f"  ⚠ Text is very long ({len(text)} chars), truncating to {max_chars} chars")
            text = text[:max_chars]
        
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
            print(f"  ✗ Error cleaning text with OpenAI: {e}")
            print("  ⚠ Falling back to raw text")
            return text
    
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
        skip_tables: bool = False,
        clean: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            skip_text: Skip text extraction
            skip_tables: Skip table extraction
            
        Returns:
            Dictionary with processing results including files created and errors
        """
        print(f"\n📄 Processing: {pdf_path.name}")
        
        base_name = pdf_path.stem
        results = {
            "pdf": pdf_path.name,
            "text_file": None,
            "table_files": [],
            "errors": []
        }
        
        # Extract and process text
        if not skip_text:
            await self._process_text(pdf_path, base_name, results, clean)
        
        # Extract and process tables
        if not skip_tables:
            await self._process_tables(pdf_path, base_name, results, clean)
        
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
    
    async def _process_tables(self, pdf_path: Path, base_name: str, results: Dict[str, Any], clean: bool = True) -> None:
        """Extract and process tables from PDF."""
        print("  📊 Extracting tables...")
        # Run synchronous pdfplumber extraction in executor
        loop = asyncio.get_running_loop()
        tables = await loop.run_in_executor(None, self.extract_tables_from_pdf, pdf_path)
        
        if not tables:
            print("  ℹ No tables found")
            return
            
        print(f"  Found {len(tables)} tables")
        
        async def process_single_table(idx: int, page_num: int, df: pd.DataFrame):
            if clean:
                # print(f"  🤖 Processing table {idx}...", end="", flush=True)
                cleaned_df = await self.process_table_with_openai(df, page_num, idx)
            else:
                cleaned_df = df
            
            if cleaned_df is not None and not cleaned_df.empty:
                csv_file = self.processed_dir / f"{base_name}_table_{idx}.csv"
                # file processing in main thread is fine for small files
                cleaned_df.to_csv(csv_file, index=False)
                results["table_files"].append(csv_file.name)
            else:
                results["errors"].append(f"Failed to process table {idx}")

        # Process tables concurrently
        if clean:
            print(f"  Processing {len(tables)} tables in parallel...")
            tasks = [process_single_table(idx, page_num, df) for idx, (page_num, df) in enumerate(tables, TABLE_INDEX_START)]
            await asyncio.gather(*tasks)
        else:
            print("  ⏩ Skipping table cleaning (saving raw CSVs)")
            for idx, (page_num, df) in enumerate(tables, TABLE_INDEX_START):
                await process_single_table(idx, page_num, df)
                
        print(f"  ✓ Saved {len(results['table_files'])} tables")
    
    async def process_all_pdfs(
        self,
        skip_text: bool = False,
        skip_tables: bool = False,
        clean: bool = True,
        file_pattern: str = "*.pdf"
    ) -> List[Dict[str, Any]]:
        """
        Process all PDF files in the raw directory.
        
        Args:
            skip_text: Skip text extraction
            skip_tables: Skip table extraction
            clean: Whether to clean text/tables with OpenAI
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
                return await self.process_pdf(pdf_path, skip_text, skip_tables, clean)

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
        total_tables = sum(len(r["table_files"]) for r in results)
        errors = sum(len(r["errors"]) for r in results)
        
        print(f"Total PDFs processed: {total_files}")
        print(f"Text files created: {text_files}")
        print(f"Table files created: {total_tables}")
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
        '--skip-tables',
        action='store_true',
        help='Skip table extraction'
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
        
        results = [await extractor.process_pdf(pdf_path, args.skip_text, args.skip_tables, clean)]
    else:
        # Process all files
        results = await extractor.process_all_pdfs(args.skip_text, args.skip_tables, clean)
    
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
