# Equinor Investor Relations Scraper

A Python-based scraper using Playwright to download financial reports from Equinor's investor relations page.

## Setup

1. Install dependencies:

```bash
uv sync
```

1. Install Playwright browsers:

```bash
uv run playwright install chromium
```

1. Configure environment variables:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your-actual-api-key-here
```

1. (Optional) Customize model settings in `config.py`:

```python
# Change models for different quality/cost tradeoffs
MODEL_EXTRACTOR = "gpt-4o"  # or "gpt-4o-mini"
MODEL_QA = "gpt-4o"         # or "gpt-4o-mini"
```

## Usage

### Basic Usage

Download all available PDFs:

```bash
uv run python scripts/scraper.py
```

### Filter by Year

Download only 2025 reports:

```bash
uv run python scripts/scraper.py --year 2025
```

Download only 2024 reports:

```bash
uv run python scripts/scraper.py --year 2024
```

### Custom Download Directory

Specify a custom download directory:

```bash
uv run python scripts/scraper.py --dir my_downloads
```

### Show Browser Window

Run with visible browser (useful for debugging):

```bash
uv run python scripts/scraper.py --no-headless
```

### Combined Options

```bash
uv run python scripts/scraper.py --year 2025 --dir reports_2025 --no-headless
```

## Features

- **Automated PDF Downloads**: Automatically finds and downloads all financial report PDFs
- **Year Filtering**: Filter reports by specific year
- **Multiple Extraction Methods**: Uses multiple strategies to find PDFs on the page
- **Error Handling**: Robust error handling with fallback download methods
- **Progress Tracking**: Shows download progress and status
- **Headless Mode**: Can run invisibly in the background or with visible browser

## Output

Downloaded PDFs are saved to the `downloads/` directory (or custom directory specified with `--dir`) with descriptive filenames based on the report titles.

## Project Structure

"""
investor_relations_scraper/
├── scripts/
│   ├── scraper.py      # Main scraper script
│   ├── extractor.py    # (other scripts)
│   └── qa_engine.py    # (other scripts)
├── downloads/          # Downloaded PDFs (created automatically)
├── pyproject.toml      # Project dependencies
└── README.md          # This file
"""

## Requirements

- Python >= 3.9
- Playwright
- aiohttp
- OpenAI API key (for extractor)

All dependencies are managed via the `pyproject.toml` file.

## PDF Extractor

The extractor processes downloaded PDFs to extract clean text and tables using OpenAI's API.

### Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Usage

Process a single PDF:

```bash
uv run python scripts/extractor.py --file "Q1-2025-report.pdf"
```

Process all PDFs in `data/raw`:

```bash
uv run python scripts/extractor.py
```

Use GPT-4o for higher quality (more expensive):

```bash
uv run python scripts/extractor.py --model gpt-4o
```

Extract only text (skip tables):

```bash
uv run python scripts/extractor.py --skip-tables
```

Extract only tables (skip text):

```bash
uv run python scripts/extractor.py --skip-text
```

### Output

The extractor creates:

- **Text files**: `data/processed/{filename}_text.txt` - Cleaned text ready for embedding
- **CSV files**: `data/processed/{filename}_table_{n}.csv` - Extracted tables

### Features

- **OpenAI-powered cleaning**: Uses GPT-4o-mini (or GPT-4o) to clean and structure extracted content
- **Smart text processing**: Removes headers, footers, page numbers while preserving meaningful content
- **Table extraction**: Detects and extracts tables with proper CSV formatting
- **Batch processing**: Process single files or entire directories
- **Error handling**: Robust error handling with fallback to raw data if API fails
