from __future__ import annotations

import logging

from pydantic import ValidationError

from llm_client import LLMClient, resolve_refs
from models import ParsedTOC, TOCPage

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a financial document analyst specializing in investment fund reports.
Your task: parse raw text from Table of Contents (TOC) pages and extract a \
structured list of all sub-funds with their page numbers, plus any shared \
report-level sections.

Context you must know:
- Each page was flagged by a heuristic algorithm as a potential TOC page.
- A "confidence" score accompanies each page — higher means stronger TOC signal, \
  lower-scored pages may be noise or unrelated content.
- Page numbers printed in the TOC refer to the document's own pagination \
  (which may differ from PDF page indices in the metadata).

Extraction rules:
1. Identify the master fund (umbrella fund) name from headers or the TOC title area.
2. Extract every sub-fund listed in the TOC with its starting page number.
3. For each sub-fund, determine its end_page: the last page before the NEXT TOC entry \
   begins — regardless of whether that next entry is another sub-fund or a generic section \
   (e.g., "Notes to the Financial Statements"). For the last sub-fund, use the last page \
   number visible in the TOC as end_page.
4. For each sub-fund, also capture named sub-sections if the TOC provides them \
   (e.g., "Statement of Net Assets", "Statement of Operations", "Schedule of Investments"). \
   For each section, determine its end_page the same way: the page before the next TOC entry \
   starts. The last section of a sub-fund ends on the sub-fund's own end_page.
5. Distinguish actual sub-fund names from generic report sections \
   ("Notes to the Financial Statements", "Report of the Board of Directors" etc. are NOT sub-funds).
6. If a sub-fund appears multiple times, keep only the first (lowest page number) occurrence.
7. Return sub-funds ordered by ascending start_page.
8. Capture report-level sections that are NOT part of any specific sub-fund into \
   shared_sections. These are consolidated financial statements or notes that cover ALL \
   sub-funds (e.g. "Statement of Financial Position", "Statement of Comprehensive Income", \
   "Statement of Changes in Net Assets", "Notes to the Financial Statements"). \
   Determine start_page and end_page for each shared section the same way as for \
   sub-fund sections. Do NOT include purely administrative sections like \
   "Directors and Other Information" or "Report of the Board of Directors".\
"""

_TOOL_NAME = "extract_toc"
_TOOL_DESCRIPTION = (
    "Submit the fully parsed Table of Contents with the master fund name "
    "and every identified sub-fund."
)


class TOCExtractor:
    """LLM-based parser that converts raw TOC page text into structured sub-fund data."""

    def __init__(self, client: LLMClient):
        self._client = client

    def extract(self, toc_pages: list[TOCPage]) -> ParsedTOC:
        """Send candidate TOC pages to the LLM and return structured TOC data."""
        if not toc_pages:
            raise ValueError("No TOC pages provided for extraction")

        logger.info(
            "Extracting TOC via %s from %d candidate pages (PDF pages: %s)",
            self._client.model,
            len(toc_pages),
            [p.page_number for p in toc_pages],
        )

        result_data = self._client.call_with_tool(
            system=_SYSTEM_PROMPT,
            user_message=self._build_user_message(toc_pages),
            tool_name=_TOOL_NAME,
            tool_description=_TOOL_DESCRIPTION,
            input_schema=resolve_refs(ParsedTOC.model_json_schema()),
        )

        try:
            result = ParsedTOC.model_validate(result_data.tool_input)
        except ValidationError:
            logger.exception("LLM response failed schema validation")
            raise

        logger.info(
            "Parsed TOC for '%s' — %d sub-funds, %d shared sections extracted",
            result.master_fund_name,
            len(result.subfunds),
            len(result.shared_sections),
        )
        return result

    def extract_from_navigator(self, navigator_results: list[dict]) -> ParsedTOC:
        """Convenience wrapper that accepts raw dict output from find_probable_toc_pages()."""
        toc_pages = [TOCPage.from_navigator_result(r) for r in navigator_results]
        return self.extract(toc_pages)

    @staticmethod
    def _build_user_message(toc_pages: list[TOCPage]) -> str:
        header = (
            "Below are candidate Table of Contents pages from a fund report PDF.\n"
            "Pages with higher confidence scores are more likely to be genuine TOC pages; "
            "lower-scored pages may be noise.\n"
        )

        page_blocks = []
        for page in toc_pages:
            page_blocks.append(
                f"--- PDF PAGE {page.page_number} (confidence: {page.confidence_score}) ---\n"
                f"{page.raw_text}"
            )

        footer = (
            "Extract the master fund name, all sub-fund entries "
            "(with their sections where available), and any shared report-level "
            "sections using the extract_toc tool."
        )

        return "\n\n".join([header, *page_blocks, footer])
