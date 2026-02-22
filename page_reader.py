from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from pypdf import PdfReader

from llm_client import LLMClient
from models import ParsedTOC, SubFundEntry

logger = logging.getLogger(__name__)

logging.getLogger("pypdf").setLevel(logging.ERROR)

_NUM_SAMPLES = 3

_PAGE_NUMBER_TOOL_NAME = "report_page_number"
_PAGE_NUMBER_TOOL_DESCRIPTION = "Report the printed page number found on this page."
_PAGE_NUMBER_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["printed_page_number"],
    "properties": {
        "printed_page_number": {
            "type": "integer",
            "description": (
                "The printed page number visible in the "
                "header or footer of this page"
            ),
        }
    },
}


class PageReader:
    """Reads specific pages from a PDF using printed page numbers.

    Calibrates the offset between printed page numbers (from the TOC)
    and 0-based PDF page indices by asking the LLM to read the printed
    page number from sample pages, then applying a majority vote.
    """

    def __init__(self, pdf_path: str, client: LLMClient):
        self.reader = PdfReader(pdf_path)
        self.total_pages = len(self.reader.pages)
        self._client = client
        self._offset: int | None = None

    def calibrate(self, toc: ParsedTOC) -> None:
        """Determine the offset between printed page numbers and PDF indices.

        Picks sample pages from across the sub-fund list, asks the LLM to
        read the printed page number from each, and takes a majority vote
        on the resulting offsets.
        """
        if not toc.subfunds:
            raise ValueError("No sub-funds in TOC to calibrate from")

        samples = self._select_samples(toc.subfunds)
        offsets: list[int] = []

        for subfund in samples:
            guess_index = subfund.start_page - 1
            if not (0 <= guess_index < self.total_pages):
                continue

            text = self.reader.pages[guess_index].extract_text() or ""
            if not text.strip():
                continue

            reported_page = self._detect_printed_page(text)
            if reported_page is None:
                logger.warning(
                    "Could not detect printed page number at PDF index %d "
                    "(sample: '%s')",
                    guess_index,
                    subfund.name,
                )
                continue

            offset = guess_index - (reported_page - 1)
            offsets.append(offset)
            logger.info(
                "Sample '%s': PDF index %d → printed page %d → offset %d",
                subfund.name,
                guess_index,
                reported_page,
                offset,
            )

        if not offsets:
            logger.warning(
                "Could not determine offset from any sample — falling back to 0"
            )
            self._offset = 0
            return

        counts = Counter(offsets)
        majority_offset, majority_count = counts.most_common(1)[0]

        if len(set(offsets)) > 1:
            logger.warning(
                "Inconsistent offsets across %d samples: %s — "
                "using majority vote: offset=%d (%d/%d agree)",
                len(offsets),
                offsets,
                majority_offset,
                majority_count,
                len(offsets),
            )
        else:
            logger.info(
                "Calibrated offset=%d (%d/%d samples agree)",
                majority_offset,
                len(offsets),
                len(offsets),
            )

        self._offset = majority_offset

    @staticmethod
    def _select_samples(subfunds: list[SubFundEntry]) -> list[SubFundEntry]:
        """Pick up to _NUM_SAMPLES sub-funds spread across the list."""
        n = len(subfunds)
        if n <= _NUM_SAMPLES:
            return list(subfunds)
        step = (n - 1) / (_NUM_SAMPLES - 1)
        indices = [round(i * step) for i in range(_NUM_SAMPLES)]
        return [subfunds[i] for i in indices]

    def _detect_printed_page(self, page_text: str) -> int | None:
        """Ask the LLM to read the printed page number from a page's text."""
        try:
            result = self._client.call_with_tool(
                system=(
                    "You receive the extracted text of a single PDF page. "
                    "Determine the printed page number as shown on the page itself "
                    "(typically found in the header or footer). "
                    "Report only the numeric page number."
                ),
                user_message=f"--- PAGE TEXT ---\n{page_text}",
                tool_name=_PAGE_NUMBER_TOOL_NAME,
                tool_description=_PAGE_NUMBER_TOOL_DESCRIPTION,
                input_schema=_PAGE_NUMBER_TOOL_SCHEMA,
                max_tokens=128,
            )
        except (ValueError, Exception):
            logger.debug("LLM failed to return a valid page number tool call")
            return None

        return result.tool_input.get("printed_page_number")

    @property
    def offset(self) -> int:
        if self._offset is None:
            raise RuntimeError("PageReader not calibrated — call calibrate() first")
        return self._offset

    def printed_to_index(self, printed_page: int) -> int:
        """Convert a printed page number to a 0-based PDF page index."""
        return printed_page - 1 + self.offset

    def get_page_text(self, printed_page: int) -> str:
        """Extract text from a single page by its printed page number."""
        idx = self.printed_to_index(printed_page)
        if 0 <= idx < self.total_pages:
            return self.reader.pages[idx].extract_text() or ""
        logger.warning(
            "Printed page %d → PDF index %d is out of range (total: %d)",
            printed_page,
            idx,
            self.total_pages,
        )
        return ""

    def get_range_text(self, start_page: int, end_page: int) -> dict[int, str]:
        """Extract text for a range of printed pages (inclusive)."""
        return {p: self.get_page_text(p) for p in range(start_page, end_page + 1)}
