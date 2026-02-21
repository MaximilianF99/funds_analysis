from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pdf_navigator import find_probable_toc_pages
from toc_extractor import TOCExtractor

logger = logging.getLogger(__name__)


def build_cli() -> argparse.ArgumentParser:
    """Define the fundextract command-line interface."""
    parser = argparse.ArgumentParser(
        prog="fundextract",
        description="Extract sub-fund data from investment fund report PDFs.",
    )
    parser.add_argument(
        "input_pdf",
        type=Path,
        help="Path to the fund report PDF",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON file path (default: print to stdout)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model identifier (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def run(pdf_path: Path, model: str) -> dict:
    """Execute the full extraction pipeline and return the result dict."""

    # Step 1 — Heuristic TOC page detection
    logger.info("Scanning PDF for Table of Contents pages …")
    toc_pages = find_probable_toc_pages(str(pdf_path))

    if not toc_pages:
        logger.error("No TOC pages detected in %s", pdf_path.name)
        sys.exit(1)

    logger.info(
        "Found %d candidate TOC pages: %s",
        len(toc_pages),
        [p["page_num"] for p in toc_pages],
    )

    # Step 2 — LLM-based TOC parsing
    extractor = TOCExtractor(model=model)
    parsed_toc = extractor.extract_from_navigator(toc_pages)

    logger.info(
        "Extracted %d sub-funds for '%s'",
        len(parsed_toc.subfunds),
        parsed_toc.master_fund_name,
    )

    # Step 3 — Per-subfund data extraction (future component)
    # TODO: DataExtractor will iterate over parsed_toc.subfunds,
    #       read their page ranges, and extract the required variables
    #       (NAV, currency, share classes, income/expenses, etc.)

    return {
        "source_file": pdf_path.name,
        "toc": parsed_toc.model_dump(mode="json"),
    }


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    if not args.input_pdf.exists():
        print(f"Error: file not found: {args.input_pdf}", file=sys.stderr)
        sys.exit(1)

    result = run(args.input_pdf, model=args.model)
    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output_json, encoding="utf-8")
        print(f"Output written to {args.out}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
