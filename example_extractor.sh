#!/bin/bash

# Example usage script for the PDF extractor

# Set your OpenAI API key (required)
export OPENAI_API_KEY="your-api-key-here"

# Example 1: Process a single PDF file
echo "Example 1: Process a single PDF file"
uv run python scripts/extractor.py --file "Q1-2025-report.pdf"

# Example 2: Process all PDFs in data/raw
echo -e "\nExample 2: Process all PDFs"
uv run python scripts/extractor.py

# Example 3: Process with GPT-4o (higher quality, more expensive)
echo -e "\nExample 3: Use GPT-4o model"
uv run python scripts/extractor.py --file "Q1-2025-report.pdf" --model gpt-4o

# Example 4: Extract only text (skip tables)
echo -e "\nExample 4: Extract only text"
uv run python scripts/extractor.py --file "Q1-2025-report.pdf" --skip-tables

# Example 5: Extract only tables (skip text)
echo -e "\nExample 5: Extract only tables"
uv run python scripts/extractor.py --file "Q1-2025-report.pdf" --skip-text

# Example 6: Custom directories
echo -e "\nExample 6: Custom input/output directories"
uv run python scripts/extractor.py --raw-dir custom/input --processed-dir custom/output
