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
uv run python examples/basic_scraper.py
# OR
uv run ir-scraper
```

### Filter by Year

Download only 2025 reports:

```bash
uv run ir-scraper --year 2025
```

Download only 2024 reports:

```bash
uv run ir-scraper --year 2024
```

### Custom Download Directory

Specify a custom download directory:

```bash
uv run ir-scraper --dir my_downloads
```

### Show Browser Window

Run with visible browser (useful for debugging):

```bash
uv run ir-scraper --no-headless
```

### Combined Options

```bash
uv run ir-scraper --year 2025 --dir reports_2025 --no-headless
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
├── src/
│   └── investor_relations_scraper/
│       ├── scraper.py    # Main scraper
│       ├── cli.py        # PDF extractor CLI
│       ├── qa_engine.py  # QA engine logic
│       └── extractors/   # Extraction strategies
├── examples/             # Example usage scripts
├── tests/                # Test suite
├── data/                 # Data directory
├── pyproject.toml        # Project dependencies
└── README.md             # This file
"""

## Requirements

- Python >= 3.9
- Playwright
- aiohttp
- OpenAI API key (for extractor)

All dependencies are managed via the `pyproject.toml` file.

## PDF Extractor

The extractor uses `pdfplumber` to process downloaded PDFs and the OpenAI API to clean and structure the data, extracting:

- Clean text content
- Structured tables (CSV)
- Metadata

### Extractor Usage

Process a single PDF:

```bash
uv run ir-extractor --file "Q1-2025-report.pdf"
```

Process all PDFs in `data/raw`:

```bash
uv run ir-extractor
```

### Extractor Output

The extractor creates:

- **Text files**: `data/processed/{filename}_text.txt`
- **Tables**: `data/processed/{filename}_table_{n}.csv`

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [**System Architecture**](docs/architecture.md): High-level design and module breakdown.
- [**Cheat Sheet**](docs/cheat_sheet.md): Quick reference for common commands.
