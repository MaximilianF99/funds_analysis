# fundextract

A CLI tool that extracts structured financial data from investment fund report PDFs using LLM-based parsing.

Built as a solution to the coding challenge: automatically extract per-sub-fund financial variables (NAV, share classes, income/expenses, reporting period) from multi-sub-fund annual reports — without hardcoded page numbers.

---

## Setup

### Requirements

- Python 3.11+
- An Anthropic API key **or** a Google Gemini API key

### Installation

```bash
pip install pypdf pydantic anthropic google-genai
```

Set your API key as an environment variable:

```bash
# For Claude (default)
export ANTHROPIC_API_KEY=sk-...

# For Gemini
export GOOGLE_API_KEY=...
```

---

## Usage

```bash
# Extract all sub-funds from a PDF, write JSON output to a file
python main.py report.pdf --out output/result.json

# Use Gemini instead of Claude
python main.py report.pdf --model gemini-2.5-flash --out output/result.json

# Extract only a specific sub-fund (substring match, case-insensitive)
python main.py report.pdf --subfund "Global Equity" --out output/result.json

# Only parse the Table of Contents (fast sanity check)
python main.py report.pdf --toc-only

# Enable verbose logging
python main.py report.pdf -v --out output/result.json
```

---

## Output Format

The tool outputs a JSON file with the following structure:

```json
{
  "source_file": "report.pdf",
  "master_fund_name": "Goldman Sachs Funds, plc",
  "subfunds": [
    {
      "subfund_name": "Goldman Sachs US$ Liquid Reserves Fund",
      "fund_currency": "USD",
      "total_nav": 33774089450.0,
      "share_classes": [
        {
          "name": "Institutional Distribution Share Class",
          "currency": "USD",
          "nav": 26708553307.0,
          "nav_per_share": 1.0,
          "outstanding_shares": 26708553374.0
        }
      ],
      "income_expenses": [
        { "name": "Interest income", "amount": 1234567.89 },
        { "name": "Management fees", "amount": -456789.01 }
      ],
      "reporting_period_start": "2023-01-01",
      "reporting_period_end": "2023-12-31",
      "source_pages": {
        "total_nav": [5],
        "share_classes": [62, 63],
        "income_expenses": [8, 9]
      },
      "missing_fields": []
    }
  ]
}
```

`source_pages` maps each field group to the **printed page numbers** in the PDF where the data was found — enabling independent verification by auditors.

---

## Extracted Variables

| Field | Description |
|---|---|
| `fund_currency` | Base currency of the sub-fund (ISO 4217) |
| `total_nav` | Total Net Asset Value of the sub-fund |
| `share_classes[].name` | Share class name (e.g. "Class A", "Class I USD") |
| `share_classes[].currency` | Currency of the share class |
| `share_classes[].nav` | Aggregate NAV for this share class |
| `share_classes[].nav_per_share` | NAV per individual share |
| `share_classes[].outstanding_shares` | Number of outstanding shares at period end |
| `income_expenses[].name` | Line item name from Statement of Operations |
| `income_expenses[].amount` | Amount (positive = income, negative = expense) |
| `reporting_period_start` | Start of the reporting period (ISO 8601) |
| `reporting_period_end` | End of the reporting period (ISO 8601) |
| `source_pages` | Page numbers per field group for audit trail |
| `missing_fields` | Fields that could not be extracted |

---

## Architecture

The tool runs a four-step pipeline:

```
PDF
 │
 ├─ Step 1 │ pdf_navigator.py     │ Heuristic TOC page detection
 │          │                      │ Scores pages by keyword presence + TOC line patterns
 │          │                      │ No LLM involved — fast pre-filter
 │
 ├─ Step 2 │ toc_extractor.py     │ LLM-based TOC parsing
 │          │                      │ LLM receives candidate pages + Pydantic schema
 │          │                      │ Returns: master fund name, all sub-funds with page ranges
 │
 ├─ Step 3 │ page_reader.py       │ Page offset calibration
 │          │                      │ Printed page numbers ≠ PDF indices
 │          │                      │ LLM reads printed number from sample pages → majority vote
 │
 └─ Step 4 │ subfund_extractor.py │ Per-sub-fund financial data extraction
            │                      │ Smart page selection (specific sections or full range)
            │                      │ Shared consolidated sections filtered by sub-fund name
            │                      │ LLM extracts all variables via forced tool use
```

### Key Design Decisions

**LLM Tool Use (Forced Function Calling)**
All LLM calls use forced tool use: the model receives the Pydantic model's JSON Schema and *must* return a structured response matching it. This eliminates free-text parsing and makes extraction reliable across varying PDF layouts.

**Provider-agnostic LLM client**
`llm_client.py` defines a `Protocol` interface (`LLMClient`) with two concrete implementations: `AnthropicClient` (Claude) and `GeminiClient`. The active provider is selected automatically by model name prefix (`claude-` / `gemini-`). Adding a new provider requires only a new adapter class and one line in the factory dict.

**No hardcoded page numbers**
Page ranges come entirely from the LLM-parsed TOC. The page offset between printed numbers and PDF indices is calibrated automatically at runtime via LLM-assisted sampling.

**Graceful degradation**
If a sub-fund cannot be extracted (LLM error, validation failure), the pipeline continues and returns a result with `missing_fields` populated — rather than aborting the entire run. TOC parsing failures *do* abort, as there is no meaningful fallback.

**Multilingual support**
Keyword sets cover English, German, and French terminology throughout (e.g. `"Vermögensaufstellung"`, `"état des actifs nets"`, `"compte de résultat"`).

---

## Assumptions & Limitations

**Assumptions**
- The PDF contains a readable Table of Contents with sub-fund names and page numbers.
- Text is extractable via pypdf (scanned image-only PDFs are not supported).
- At least one of the three TOC keyword languages (EN/DE/FR) appears in the TOC heading.
- The LLM API is reachable and the API key is valid.

**Known Limitations**
- **Scanned PDFs**: pypdf cannot extract text from image-based PDFs. An OCR preprocessing step (e.g. pytesseract) would be required.
- **Inconsistent page offsets**: Some PDFs use different offset conventions per chapter. The majority-vote calibration handles most cases but may fail on unusual layouts.
- **Very long shared sections**: Shared consolidated sections >5 pages are filtered by sub-fund name occurrence. If a sub-fund's data appears on a page that doesn't mention its name, those pages are missed.
- **Non-standard TOC formats**: Highly non-standard or missing TOCs will cause Step 1/2 to fail.
- **Cost**: Each sub-fund requires one LLM call. Large funds with many sub-funds (50+) will incur proportionally higher API costs.

---

## Project Structure

```
funds_analyse/
├── main.py                  # CLI entry point + pipeline orchestration
├── models.py                # Pydantic data models (TOC + extraction output)
├── pdf_navigator.py         # Step 1: heuristic TOC page detection
├── toc_extractor.py         # Step 2: LLM-based TOC parsing
├── page_reader.py           # Step 3: page offset calibration + text retrieval
├── subfund_extractor.py     # Step 4: per-sub-fund data extraction
├── llm_client.py            # LLM adapter (Anthropic + Gemini) + schema utilities
└── output/                  # Example outputs (JSON per PDF)
```

---
