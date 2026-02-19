# Scraper Quick Reference

## Setup

```bash
uv sync
cp .env.example .env
# Edit .env with your OpenAI API Key
```

## Basic Commands

### Scraper (Download PDFs)

```bash
# Download all
uv run ir-scraper

# Download specific year
uv run ir-scraper --year 2025
```

### Extractor (Process PDFs)

```bash
# Process all downloaded PDFs
uv run ir-extractor

# Process a single specific file
uv run ir-extractor --file "Q1-2025-report.pdf"

# Skip cleaning (faster, raw output only)
uv run ir-extractor --no-cleaning
```

### QA Chat

```bash
uv run python examples/chat_qa.py
```

## Extraction Methods

Set `PDF_EXTRACTION_METHOD` in `config.py`:

| Value | Description |
|-------|-------------|
| `"pdfplumber"` | Fast, text-based (digital PDFs only) |
| `"qwen-vl"` | Full OCR via Qwen2.5-VL |
| `"fallback"` (default) | pdfplumber + OCR fallback per-page |

## Common Options

### Extractor

| Option | Description |
|--------|-------------|
| `--file FILENAME` | Process specific file |
| `--skip-text` | Skip text extraction |
| `--skip-tables` | Skip table extraction |
| `--no-cleaning` | Skip OpenAI text cleaning |
| `--raw-dir DIR` | Custom input directory |
| `--processed-dir DIR` | Custom output directory |

## Output Files

| File | Content |
|------|---------|
| `{name}_text.txt` | Clean text content |
| `{name}_table_{n}.csv` | Extracted tables in CSV format |

## Troubleshooting

### OpenAI API Error

```bash
# Verify API key in .env
cat .env
```

### File Not Found

Ensure the filename passed to `--file` exists in `data/raw/`.

## Performance Tips

- **Text Only**: Use `--skip-tables` for much faster processing if you only need text search.
- **No Cleaning**: Use `--no-cleaning` to skip OpenAI processing for raw extraction.
- **Process Single File**: Always test with one file first before running the full batch.

## Documentation

- [**System Architecture**](architecture.md): High-level system design.
- [**README**](../README.md): Main project guide.
