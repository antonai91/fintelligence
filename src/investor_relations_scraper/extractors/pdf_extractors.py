"""
Concrete implementations of PDF extraction strategies.

This module provides different strategies for extracting text and tables from PDFs:
- PdfPlumberExtractor: Fast text-based extraction using pdfplumber
- Qwen25VLExtractor: OCR-based extraction using Qwen2.5-VL-3B-Instruct-GGUF
"""

from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import pdfplumber

from .base import BasePDFExtractor


class PdfPlumberExtractor(BasePDFExtractor):
    """Extract text and tables from PDFs using pdfplumber (fast, good for text-based PDFs)."""
    
    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract raw text from PDF file using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
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
        """
        Extract tables from PDF file using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of tuples (page_number, DataFrame)
        """
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()
                    
                    for table_data in page_tables:
                        if table_data and len(table_data) > 0:
                            # Convert to DataFrame
                            df = pd.DataFrame(table_data)
                            
                            # Skip empty tables
                            if not df.empty and df.shape[0] > 1:
                                tables.append((page_num, df))
                                
            return tables
        except Exception as e:
            print(f"✗ Error extracting tables from {pdf_path.name}: {str(e)}")
            return []
    
    def supports_ocr(self) -> bool:
        """pdfplumber does not use OCR."""
        return False
    
    def get_name(self) -> str:
        """Get the name of this extractor."""
        return "pdfplumber"


class Qwen25VLExtractor(BasePDFExtractor):
    """
    Extract text from PDFs using Qwen2.5-VL-3B-Instruct-GGUF model.
    
    This extractor converts PDF pages to images and uses a vision-language model
    for OCR. It's better for scanned PDFs and complex layouts, but slower than pdfplumber.
    """
    
    def __init__(
        self,
        model_path: str = "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
        model_file: str = "qwen2_5-vl-3b-instruct-q4_k_m.gguf",
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        verbose: bool = False
    ):
        """
        Initialize the Qwen2.5-VL extractor.
        
        Args:
            model_path: HuggingFace model ID or local path
            model_file: Specific GGUF file to use
            n_gpu_layers: Number of layers to offload to GPU (-1 for all, uses Metal on Mac)
            n_ctx: Context window size
            verbose: Whether to print verbose llama.cpp output
        """
        self.model_path = model_path
        self.model_file = model_file
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        self._model = None
        self._initialized = False
    
    def _initialize_model(self):
        """Lazy initialization of the model (only load when needed)."""
        if self._initialized:
            return
        
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            
            print(f"Loading Qwen2.5-VL model from {self.model_path}...")
            print(f"  Model file: {self.model_file}")
            print(f"  GPU layers: {self.n_gpu_layers} (Metal acceleration on Mac)")
            
            # Initialize the model with vision support
            self._model = Llama.from_pretrained(
                repo_id=self.model_path,
                filename=self.model_file,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=self.verbose,
                chat_format="llava-1-5"  # Use LLaVA chat format for vision models
            )
            
            self._initialized = True
            print("✓ Model loaded successfully")
            
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for Qwen2.5-VL extraction. "
                "Install it with: pip install llama-cpp-python"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qwen2.5-VL model: {e}")
    
    def _pdf_to_images(self, pdf_path: Path) -> List[Tuple[int, any]]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of tuples (page_number, PIL.Image)
        """
        try:
            from pdf2image import convert_from_path
            
            print(f"  Converting PDF to images...")
            images = convert_from_path(pdf_path)
            return [(i + 1, img) for i, img in enumerate(images)]
            
        except ImportError:
            raise ImportError(
                "pdf2image is required for Qwen2.5-VL extraction. "
                "Install it with: pip install pdf2image"
            )
        except Exception as e:
            print(f"✗ Error converting PDF to images: {e}")
            return []
    
    def _extract_text_from_image(self, image: any, page_num: int) -> str:
        """
        Extract text from a single image using Qwen2.5-VL.
        
        Args:
            image: PIL Image
            page_num: Page number for logging
            
        Returns:
            Extracted text
        """
        import base64
        from io import BytesIO
        
        try:
            # Convert image to base64 for llama.cpp
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            data_uri = f"data:image/png;base64,{img_base64}"
            
            # Create prompt for OCR
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "Extract all text from this image. Preserve the layout and structure as much as possible. Return only the extracted text, no explanations."}
                    ]
                }
            ]
            
            # Generate response
            response = self._model.create_chat_completion(
                messages=messages,
                max_tokens=2048,
                temperature=0.0
            )
            
            extracted_text = response["choices"][0]["message"]["content"]
            return extracted_text
            
        except Exception as e:
            print(f"  ✗ Error extracting text from page {page_num}: {e}")
            return ""
    
    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using Qwen2.5-VL OCR.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        self._initialize_model()
        
        # Convert PDF to images
        images = self._pdf_to_images(pdf_path)
        if not images:
            return ""
        
        print(f"  Processing {len(images)} pages with Qwen2.5-VL...")
        text_content = []
        
        for page_num, image in images:
            print(f"  Processing page {page_num}/{len(images)}...")
            text = self._extract_text_from_image(image, page_num)
            if text:
                text_content.append(f"--- Page {page_num} ---\n{text}")
        
        return "\n\n".join(text_content)
    
    def extract_tables(self, pdf_path: Path) -> List[Tuple[int, pd.DataFrame]]:
        """
        Extract tables from PDF using Qwen2.5-VL.
        
        Note: This is a simplified implementation. For production use,
        you might want to add specific table extraction prompts.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of tuples (page_number, DataFrame)
        """
        # For now, we'll return empty list as table extraction with VLM
        # requires more sophisticated prompting and parsing
        # This can be enhanced later if needed
        print("  Note: Table extraction not yet implemented for Qwen2.5-VL")
        return []
    
    def supports_ocr(self) -> bool:
        """Qwen2.5-VL uses OCR."""
        return True
    
    def get_name(self) -> str:
        """Get the name of this extractor."""
        return "Qwen2.5-VL-3B-Instruct-GGUF"
