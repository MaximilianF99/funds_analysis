from __future__ import annotations

import logging
from typing import Any

from anthropic import Anthropic
from pydantic import ValidationError

from llm_utils import resolve_refs
from models import SubFundEntry, SubFundResult, TOCSection
from page_reader import PageReader

logger = logging.getLogger(__name__)

_MAX_UNFILTERED_PAGES = 5

_SECTION_KEYWORDS_NAV: frozenset[str] = frozenset({
    "net assets",
    "net asset value",
    "nav",
    "financial position",
    "assets and liabilities",
    "balance sheet",
    "statement of assets",
    "statement of net assets",
    "vermögensaufstellung",
    "état des actifs nets",
    "bilan",
})

_SECTION_KEYWORDS_INCOME: frozenset[str] = frozenset({
    "operations",
    "comprehensive income",
    "income and expenditure",
    "income and expenses",
    "profit and loss",
    "profit or loss",
    "revenue",
    "operating results",
    "statement of operations",
    "ertrags- und aufwandsrechnung",
    "compte de résultat",
})

_SECTION_KEYWORDS_SHARES: frozenset[str] = frozenset({
    "share capital",
    "shares outstanding",
    "shares in issue",
    "number of shares",
    "share class",
    "participating shares",
    "redeemable shares",
    "changes in net assets",
    "changes in shares",
    "capital activity",
    "statistiques",
    "développement des parts",
})

_SECTION_KEYWORDS_NOTES: frozenset[str] = frozenset({
    "notes to the financial",
    "notes to financial",
    "notes",
    "anhang",
    "annexe",
})

_ALL_RELEVANT_KEYWORDS: frozenset[str] = (
    _SECTION_KEYWORDS_NAV
    | _SECTION_KEYWORDS_INCOME
    | _SECTION_KEYWORDS_SHARES
    | _SECTION_KEYWORDS_NOTES
)

_SYSTEM_PROMPT = """\
You are a financial document analyst specializing in investment fund reports.
You receive pages from an annual or semi-annual fund report PDF. The pages may \
include both sub-fund-specific sections AND consolidated/shared report sections \
that cover multiple sub-funds (e.g. a combined Statement of Financial Position \
with columns per sub-fund, or Notes to the Financial Statements).

IMPORTANT: You must extract data ONLY for the sub-fund named in the request. \
Consolidated tables often show all sub-funds side by side — pick only the column \
or rows belonging to the target sub-fund.

Extract the following data points:

1. **fund_currency** — The base currency of the sub-fund (ISO 4217).
2. **total_nav** — The total Net Asset Value of the sub-fund.
3. **share_classes** — For every share class listed:
   - name, currency, NAV, and number of outstanding shares at period end.
4. **income_expenses** — Every income and expense line item from the \
Statement of Operations / Statement of Comprehensive Income (or equivalent). \
Use positive amounts for income, negative for expenses.
5. **reporting_period_start** and **reporting_period_end** — As ISO 8601 dates.
6. **source_pages** — For each field group you populate, list the printed page \
numbers where you found the data (e.g. {"total_nav": [5], "share_classes": [62, 63]}).
7. **missing_fields** — If any of the above cannot be determined from the \
provided text, list the field names here.

Extraction guidelines:
- Page numbers in headers/footers are *printed* page numbers — report those.
- If the document uses thousands separators (commas or dots), parse them correctly.
- Currency codes should be ISO 4217 (USD, EUR, GBP, CHF, JPY, etc.).
- Do NOT invent data. If a value is not present, mark it as missing.
- When reading consolidated tables, carefully identify which column belongs to \
  the target sub-fund by matching the fund name in the column header.\
"""


class SubFundExtractor:
    """Extracts structured financial data from sub-fund pages via a single LLM call."""

    def __init__(self, page_reader: PageReader, model: str = "claude-sonnet-4-6"):
        self.page_reader = page_reader
        self.client = Anthropic()
        self.model = model

    def extract(
        self,
        entry: SubFundEntry,
        shared_sections: list[TOCSection] | None = None,
    ) -> SubFundResult:
        """Extract all required variables for a single sub-fund."""
        pages = self._collect_pages(entry, shared_sections or [])
        if not pages:
            logger.warning("No page text available for '%s'", entry.name)
            return SubFundResult(
                subfund_name=entry.name,
                missing_fields=[
                    "fund_currency", "total_nav", "share_classes",
                    "income_expenses", "reporting_period_start", "reporting_period_end",
                ],
            )

        user_message = self._build_user_message(entry.name, pages)

        logger.info(
            "Extracting data for '%s' (pages %d–%d, %d pages of text, "
            "page numbers: %s)",
            entry.name,
            entry.start_page,
            entry.end_page,
            len(pages),
            sorted(pages.keys()),
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            tools=[self._tool_definition()],
            tool_choice={"type": "tool", "name": "extract_subfund_data"},
        )

        result = self._parse_response(response, entry.name)
        logger.info(
            "Extracted data for '%s' — NAV=%s, %d share classes, %d income/expense items, "
            "missing: %s",
            result.subfund_name,
            result.total_nav,
            len(result.share_classes),
            len(result.income_expenses),
            result.missing_fields or "none",
        )
        return result

    def extract_all(
        self,
        entries: list[SubFundEntry],
        shared_sections: list[TOCSection] | None = None,
    ) -> list[SubFundResult]:
        """Extract data for all sub-funds sequentially."""
        results: list[SubFundResult] = []
        for i, entry in enumerate(entries, 1):
            logger.info("Processing sub-fund %d/%d: '%s'", i, len(entries), entry.name)
            results.append(self.extract(entry, shared_sections))
        return results

    def _collect_pages(
        self,
        entry: SubFundEntry,
        shared_sections: list[TOCSection],
    ) -> dict[int, str]:
        """Gather page text from the sub-fund's own sections and relevant shared sections."""
        pages: dict[int, str] = {}

        subfund_nav = self._find_section(entry.sections, _SECTION_KEYWORDS_NAV)
        subfund_income = self._find_section(entry.sections, _SECTION_KEYWORDS_INCOME)

        if subfund_nav or subfund_income:
            if subfund_nav:
                pages.update(self.page_reader.get_range_text(
                    subfund_nav.start_page, subfund_nav.end_page,
                ))
            if subfund_income:
                pages.update(self.page_reader.get_range_text(
                    subfund_income.start_page, subfund_income.end_page,
                ))
        else:
            pages.update(self.page_reader.get_range_text(
                entry.start_page, entry.end_page,
            ))

        shared_pages = self._collect_shared_pages(entry.name, shared_sections)
        pages.update(shared_pages)

        return pages

    def _collect_shared_pages(
        self,
        subfund_name: str,
        shared_sections: list[TOCSection],
    ) -> dict[int, str]:
        """Collect pages from shared report sections that are relevant to this sub-fund."""
        pages: dict[int, str] = {}

        for section in shared_sections:
            if not self._is_relevant_shared_section(section.title):
                continue

            section_pages = self.page_reader.get_range_text(
                section.start_page, section.end_page,
            )
            section_length = section.end_page - section.start_page + 1

            if section_length <= _MAX_UNFILTERED_PAGES:
                pages.update(section_pages)
                logger.debug(
                    "Including full shared section '%s' (%d pages) for '%s'",
                    section.title,
                    section_length,
                    subfund_name,
                )
            else:
                filtered = self._filter_pages_by_subfund_name(
                    section_pages, subfund_name,
                )
                pages.update(filtered)
                logger.debug(
                    "Filtered shared section '%s' (%d → %d pages) for '%s'",
                    section.title,
                    section_length,
                    len(filtered),
                    subfund_name,
                )

        return pages

    @staticmethod
    def _is_relevant_shared_section(title: str) -> bool:
        title_lower = title.lower()
        return any(kw in title_lower for kw in _ALL_RELEVANT_KEYWORDS)

    @staticmethod
    def _filter_pages_by_subfund_name(
        pages: dict[int, str],
        subfund_name: str,
    ) -> dict[int, str]:
        """Keep only pages that mention the sub-fund by name (case-insensitive)."""
        needle = subfund_name.lower()
        tokens = needle.split()
        short_name = " ".join(tokens[-3:]) if len(tokens) > 3 else needle

        return {
            page_num: text
            for page_num, text in pages.items()
            if needle in text.lower() or short_name in text.lower()
        }

    @staticmethod
    def _find_section(
        sections: list[TOCSection],
        keywords: frozenset[str],
    ) -> TOCSection | None:
        """Find the first section whose title matches any of the keywords."""
        for section in sections:
            title_lower = section.title.lower()
            if any(kw in title_lower for kw in keywords):
                return section
        return None

    @staticmethod
    def _build_user_message(subfund_name: str, pages: dict[int, str]) -> str:
        header = (
            f"Sub-fund: **{subfund_name}**\n\n"
            "Below are pages from the fund report that may contain data for this sub-fund.\n"
            "Some pages may be from shared/consolidated sections covering multiple sub-funds — "
            "extract ONLY the data belonging to the sub-fund named above.\n"
        )

        page_blocks = []
        for page_num in sorted(pages):
            text = pages[page_num]
            if text.strip():
                page_blocks.append(
                    f"--- PRINTED PAGE {page_num} ---\n{text}"
                )

        footer = (
            f"Extract all available financial data for **{subfund_name}** "
            "using the extract_subfund_data tool."
        )

        return "\n\n".join([header, *page_blocks, footer])

    @staticmethod
    def _tool_definition() -> dict[str, Any]:
        return {
            "name": "extract_subfund_data",
            "description": (
                "Submit the extracted financial data for a single sub-fund, "
                "including NAV, share classes, income/expenses, and reporting period."
            ),
            "input_schema": resolve_refs(SubFundResult.model_json_schema()),
        }

    @staticmethod
    def _parse_response(response: Any, subfund_name: str) -> SubFundResult:
        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_subfund_data":
                try:
                    return SubFundResult.model_validate(block.input)
                except ValidationError:
                    logger.exception(
                        "LLM response for '%s' failed schema validation", subfund_name
                    )
                    raise

        raise ValueError(
            f"No extract_subfund_data tool call in LLM response for '{subfund_name}' "
            f"(stop_reason={response.stop_reason})"
        )
