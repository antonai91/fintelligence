"""
Concrete implementations of PDF extraction strategies.

This module provides different strategies for extracting text and tables from PDFs:
- PdfPlumberExtractor: Fast text-based extraction using pdfplumber
- OllamaVisionExtractor: Vision-based table extraction using local Ollama
"""

import base64
import re
from io import BytesIO, StringIO
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import pdfplumber

from .base import BasePDFExtractor


class PdfPlumberExtractor(BasePDFExtractor):
    """Extract text and tables from PDFs using pdfplumber (fast, good for text-based PDFs)."""

    def extract_text(self, pdf_path: Path) -> str:
        """Extract raw text from PDF using pdfplumber."""
        text_content = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_content.append(f"--- Page {page_num} ---\n{text}")
            return "\n\n".join(text_content)
        except Exception as e:
            print(f"✗ Error extracting text from {pdf_path.name}: {str(e)}")
            return ""

    def extract_tables(self, pdf_path: Path) -> List[Tuple[int, pd.DataFrame]]:
        """Extract tables from PDF using pdfplumber, using first row as headers."""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    for table_data in page.extract_tables():
                        if table_data and len(table_data) > 1:
                            raw_headers = table_data[0]
                            headers = [
                                str(h).strip() if h else f"Col{i+1}"
                                for i, h in enumerate(raw_headers)
                            ]
                            # Deduplicate headers
                            seen: dict = {}
                            for i, h in enumerate(headers):
                                if h in seen:
                                    seen[h] += 1
                                    headers[i] = f"{h}_{seen[h]}"
                                else:
                                    seen[h] = 0
                            rows = [
                                [str(c).strip() if c else "" for c in row]
                                for row in table_data[1:]
                            ]
                            df = pd.DataFrame(rows, columns=headers)
                            if not df.empty:
                                tables.append((page_num, df))
            return tables
        except Exception as e:
            print(f"✗ Error extracting tables from {pdf_path.name}: {str(e)}")
            return []

    def supports_ocr(self) -> bool:
        return False

    def get_name(self) -> str:
        return "pdfplumber"


class OllamaVisionExtractor(BasePDFExtractor):
    """
    Extract tables from PDFs using a local Ollama vision model.

    Renders each PDF page as an image and prompts Ollama (e.g., llava-phi3) 
    to extract all tables as properly formatted CSV.
    Text extraction still uses pdfplumber (fast and accurate for digital PDFs).
    """

    def __init__(self, api_key: Optional[str] = "ollama", model: str = "llava-phi3"):
        """
        Args:
            api_key: Dummy key for OpenAI client
            model: Ollama model to use for vision
        """
        self._api_key = api_key
        self.model = model
        self._client = None
        self._pdfplumber = PdfPlumberExtractor()

    @property
    def client(self):
        """Lazy OpenAI client initialization."""
        if self._client is None:
            from openai import OpenAI
            from .. import config  # config is in parent package, not extractors/
            self._client = OpenAI(
                base_url=getattr(config, "OLLAMA_API_BASE", "http://localhost:11434/v1"),
                api_key=self._api_key or "ollama"
            )
        return self._client

    def _pdf_pages_to_images(self, pdf_path: Path) -> List[Tuple[int, str]]:
        """
        Convert PDF pages to base64-encoded PNG images.

        Returns:
            List of (page_num, base64_data_uri) tuples
        """
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, dpi=150)
            result = []
            for i, img in enumerate(images, 1):
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                result.append((i, f"data:image/png;base64,{b64}"))
            return result
        except ImportError:
            raise ImportError(
                "pdf2image is required for vision extraction. "
                "Install it with: uv add pdf2image"
            )
        except Exception as e:
            print(f"✗ Error converting {pdf_path.name} to images: {e}")
            return []

    def _extract_tables_from_image(self, data_uri: str, page_num: int) -> List[pd.DataFrame]:
        """Send a page image to Ollama and parse any tables as DataFrames."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                            {"type": "text", "text": (
                                "Extract ALL tables from this financial report page.\n"
                                "For each table output it as CSV:\n"
                                "- Use the exact column headers from the document\n"
                                "- Keep the row label/description in the first column\n"
                                "- Preserve all numbers exactly as shown\n"
                                "- For merged cells, repeat the value in each affected cell\n"
                                "- Remove any footnote markers (*, **, 1, 2 etc.) from cells\n"
                                "If multiple tables exist, separate them with a line containing only: ---TABLE---\n"
                                "If NO tables exist on this page, respond with exactly: NO_TABLES\n"
                                "Return ONLY the CSV data, no explanations, no markdown fences."
                            )}
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0.0,
            )

            content = response.choices[0].message.content.strip()
            if not content or content == "NO_TABLES":
                return []

            # Strip any accidental markdown fences
            content = re.sub(r'^```(?:csv)?\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)

            tables = []
            for block in re.split(r'\n\s*---TABLE---\s*\n', content):
                block = block.strip()
                if not block:
                    continue
                try:
                    df = pd.read_csv(StringIO(block))
                    if not df.empty:
                        tables.append(df)
                except Exception:
                    # Try salvaging by stripping bad lines
                    lines = [l for l in block.splitlines() if l.strip()]
                    if len(lines) >= 2:
                        try:
                            df = pd.read_csv(StringIO("\n".join(lines)))
                            if not df.empty:
                                tables.append(df)
                        except Exception:
                            pass
            return tables

        except Exception as e:
            print(f"  ✗ Error extracting tables from page {page_num}: {e}")
            return []

    def extract_text(self, pdf_path: Path) -> str:
        """Use pdfplumber for text (fast, accurate for digital PDFs)."""
        return self._pdfplumber.extract_text(pdf_path)

    def extract_tables(self, pdf_path: Path, max_workers: Optional[int] = None) -> List[Tuple[int, pd.DataFrame]]:
        """
        Extract tables using Ollama vision — processes all pages in parallel.

        Args:
            pdf_path: Path to PDF
            max_workers: Max concurrent API calls (default from config)
        """
        from .. import config
        max_workers = max_workers or getattr(config, "OLLAMA_VISION_MAX_WORKERS", 30)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        images = self._pdf_pages_to_images(pdf_path)
        if not images:
            return []

        print(f"  Extracting tables from {len(images)} pages with Ollama vision (parallel, {max_workers} workers)...")
        all_tables: List[Tuple[int, pd.DataFrame]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {
                executor.submit(self._extract_tables_from_image, data_uri, page_num): page_num
                for page_num, data_uri in images
            }
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_tables = future.result()
                    if page_tables:
                        print(f"  ✓ Page {page_num}: found {len(page_tables)} table(s)")
                        for df in page_tables:
                            all_tables.append((page_num, df))
                except Exception as e:
                    print(f"  ✗ Page {page_num} failed: {e}")

        # Sort by page number for consistent ordering
        all_tables.sort(key=lambda t: t[0])
        print(f"  📊 Total tables extracted: {len(all_tables)}")
        return all_tables

    def supports_ocr(self) -> bool:
        return True

    def get_name(self) -> str:
        return f"Ollama Vision ({self.model})"
