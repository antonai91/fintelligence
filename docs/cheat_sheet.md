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
uv run python scripts/scraper.py

# Download specific year
uv run python scripts/scraper.py --year 2025
```

### Extractor (Process PDFs)

```bash
# Process all downloaded PDFs
uv run python scripts/extractor.py

# Process a single specific file
uv run python scripts/extractor.py --file "Q1-2025-report.pdf"
```

## Common Options

### Extractor

| Option | Description |
|--------|-------------|
| `--file FILENAME` | Process specific file |
| `--skip-text` | Skip text extraction |
| `--skip-tables` | Skip table extraction |
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
- **Process Single File**: Always test with one file first before running the full batch.

## Documentation

- [**System Architecture**](architecture.md): High-level system design.
- [**README**](../README.md): Main project guide.
