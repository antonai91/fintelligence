"""
Concrete implementations of PDF extraction strategies.

This module provides different strategies for extracting text and tables from PDFs:
- PdfPlumberExtractor: Fast text-based extraction using pdfplumber
- GPT4VisionExtractor: Vision-based table extraction using OpenAI GPT-4o-mini
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


class GPT4VisionExtractor(BasePDFExtractor):
    """
    Extract tables from PDFs using GPT-4o-mini vision via the OpenAI API.

    Renders each PDF page as an image and prompts GPT-4o-mini to extract all
    tables as properly formatted CSV. Text extraction still uses pdfplumber
    (fast and accurate for digital PDFs). Uses a pdfplumber pre-filter to
    skip pages without structural tables before sending to the API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Args:
            api_key: OpenAI API key (defaults to config.OPENAI_API_KEY)
            model: OpenAI vision model to use (default: gpt-4o-mini)
        """
        from .. import config
        self._api_key = api_key or config.OPENAI_API_KEY
        self.model = model
        self._client = None
        self._pdfplumber = PdfPlumberExtractor()

    @property
    def client(self):
        """Lazy OpenAI client initialization."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _get_pages_with_tables(self, pdf_path: Path) -> set:
        """
        Fast heuristic pass: use pdfplumber to identify pages containing
        structural tables. Only those pages are sent to the vision model.
        """
        suspect_pages = set()
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    if page.extract_tables():
                        suspect_pages.add(page_num)
            return suspect_pages
        except Exception as e:
            print(f"  ⚠ Pre-filter failed for {pdf_path.name}: {e}. Defaulting to all pages.")
            return set()

    def _pdf_pages_to_images(self, pdf_path: Path) -> List[Tuple[int, str]]:
        """
        Convert PDF pages to base64-encoded PNG data URIs.

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
        """Send a single page image to GPT-4o-mini and parse returned tables as DataFrames."""
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
                    lines = [ln for ln in block.splitlines() if ln.strip()]
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

    def extract_tables(self, pdf_path: Path) -> List[Tuple[int, pd.DataFrame]]:
        """
        Extract tables from all pages using GPT-4o-mini vision.
        Uses pdfplumber pre-filter to skip pages with no structural tables.
        """
        images = self._pdf_pages_to_images(pdf_path)
        if not images:
            return []

        print("  🔍 Running fast table heuristic pre-filter...")
        suspect_pages = self._get_pages_with_tables(pdf_path)

        if not suspect_pages:
            print("  ⏩ No structural tables detected in PDF. Skipping vision pass.")
            return []

        filtered_images = [(p, uri) for p, uri in images if p in suspect_pages]
        if not filtered_images:
            filtered_images = images

        print(f"  🎯 Filtered to {len(filtered_images)} pages (out of {len(images)} total).")
        print(f"  🧠 Extracting tables with {self.model} vision...")

        all_tables: List[Tuple[int, pd.DataFrame]] = []
        for page_num, data_uri in filtered_images:
            page_tables = self._extract_tables_from_image(data_uri, page_num)
            if page_tables:
                print(f"  ✓ Page {page_num}: found {len(page_tables)} table(s)")
                for df in page_tables:
                    all_tables.append((page_num, df))

        all_tables.sort(key=lambda t: t[0])
        print(f"  📊 Total tables extracted: {len(all_tables)}")
        return all_tables

    def supports_ocr(self) -> bool:
        return True

    def get_name(self) -> str:
        return f"GPT-4o Vision ({self.model})"


class GLMOCRExtractor(BasePDFExtractor):
    """
    Extract tables from PDFs using GLM-OCR-mlx locally via mlx-vlm.

    Runs entirely on Apple Silicon (MLX) — no API calls, no cost.
    GLM-OCR is a 0.9B-parameter multimodal OCR model optimised for
    document understanding including table recognition.

    The model is loaded lazily on first use and cached for the
    lifetime of the process.
    """

    def __init__(self, model_id: Optional[str] = None, max_tokens: int = 8192):
        """
        Args:
            model_id: HuggingFace model ID (default: config.MODEL_TABLE_EXTRACTOR_LOCAL)
            max_tokens: Maximum tokens for generation (default: 8192)
        """
        from .. import config
        self.model_id = model_id or config.MODEL_TABLE_EXTRACTOR_LOCAL
        self.max_tokens = max_tokens
        self._model = None
        self._processor = None
        self._config = None
        self._pdfplumber = PdfPlumberExtractor()

    def _ensure_model_loaded(self):
        """Lazy-load the GLM-OCR model and processor on first use."""
        if self._model is None:
            print(f"  ⏳ Loading GLM-OCR model ({self.model_id})…")
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
            self._model, self._processor = load(self.model_id)
            self._config = load_config(self.model_id)
            print(f"  ✅ GLM-OCR model loaded.")

    def _page_to_pil(self, pdf_path: Path, page_num: int):
        """Convert a single PDF page to a PIL Image."""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(
                pdf_path, dpi=150, first_page=page_num, last_page=page_num
            )
            return images[0] if images else None
        except ImportError:
            raise ImportError(
                "pdf2image is required for GLM-OCR extraction. "
                "Install it with: uv add pdf2image"
            )
        except Exception as e:
            print(f"  ✗ Error converting page {page_num} to image: {e}")
            return None

    def _extract_tables_from_image(self, image, page_num: int) -> List[pd.DataFrame]:
        """Send a single page image to GLM-OCR and parse returned HTML tables."""
        try:
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template

            # Save PIL image to a temp file for mlx-vlm
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp, format="PNG")
                tmp_path = tmp.name

            prompt = "Table Recognition:"
            formatted_prompt = apply_chat_template(
                self._processor, self._config, prompt, num_images=1
            )

            output = generate(
                self._model,
                self._processor,
                formatted_prompt,
                [tmp_path],
                max_tokens=self.max_tokens,
                temperature=0.0,
                verbose=False,
            )

            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

            # mlx_vlm.generate() returns a GenerationResult dataclass — get the text
            text = output.text if hasattr(output, 'text') else str(output)

            if not text or not text.strip():
                return []

            # GLM-OCR returns HTML table markup — parse with pandas
            return self._parse_html_tables(text, page_num)

        except Exception as e:
            print(f"  ✗ GLM-OCR error on page {page_num}: {e}")
            return []

    def _parse_html_tables(self, html_content: str, page_num: int) -> List[pd.DataFrame]:
        """Parse HTML table markup from GLM-OCR output into DataFrames."""
        tables = []
        try:
            # pandas.read_html expects a string containing <table> tags
            if "<table" not in html_content.lower():
                return []
            dfs = pd.read_html(StringIO(html_content))
            for df in dfs:
                if not df.empty:
                    tables.append(df)
        except Exception as e:
            print(f"  ⚠ Could not parse GLM-OCR HTML output for page {page_num}: {e}")
        return tables

    def extract_text(self, pdf_path: Path) -> str:
        """Use pdfplumber for text (fast, accurate for digital PDFs)."""
        return self._pdfplumber.extract_text(pdf_path)

    def extract_tables(self, pdf_path: Path) -> List[Tuple[int, pd.DataFrame]]:
        """
        Extract tables from PDF pages using GLM-OCR locally.
        Uses pdfplumber pre-filter to skip pages without structural tables.
        """
        self._ensure_model_loaded()

        # Pre-filter: find pages with tables
        print("  🔍 Running fast table heuristic pre-filter...")
        suspect_pages = set()
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    if page.extract_tables():
                        suspect_pages.add(page_num)
        except Exception as e:
            print(f"  ⚠ Pre-filter failed: {e}. Scanning all pages.")
            total_pages = 0

        if not suspect_pages:
            print("  ⏩ No structural tables detected. Skipping GLM-OCR pass.")
            return []

        print(f"  🎯 Filtered to {len(suspect_pages)} pages (out of {total_pages} total).")
        print(f"  🧠 Extracting tables with GLM-OCR ({self.model_id})…")

        all_tables: List[Tuple[int, pd.DataFrame]] = []
        for page_num in sorted(suspect_pages):
            image = self._page_to_pil(pdf_path, page_num)
            if image is None:
                continue
            page_tables = self._extract_tables_from_image(image, page_num)
            if page_tables:
                print(f"  ✓ Page {page_num}: found {len(page_tables)} table(s)")
                for df in page_tables:
                    all_tables.append((page_num, df))

        print(f"  📊 Total tables extracted: {len(all_tables)}")
        return all_tables

    def supports_ocr(self) -> bool:
        return True

    def get_name(self) -> str:
        return f"GLM-OCR Local ({self.model_id})"
