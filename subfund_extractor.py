from __future__ import annotations

import logging
from typing import Any

from anthropic import Anthropic
from pydantic import ValidationError

from llm_utils import resolve_refs
from models import SubFundEntry, SubFundResult
from page_reader import PageReader

logger = logging.getLogger(__name__)

_NET_ASSETS_KEYWORDS = frozenset({
    "net assets", "assets and liabilities", "balance sheet", "financial position",
})
_OPERATIONS_KEYWORDS = frozenset({
    "operations", "income", "expenditure", "profit", "loss",
})

_SYSTEM_PROMPT = """\
You are a financial document analyst specializing in investment fund reports.
You receive the full text of all pages belonging to a single sub-fund from an \
annual or semi-annual report PDF.

Your task: extract the following data points for this sub-fund.

1. **fund_currency** — The base currency of the sub-fund (ISO 4217).
2. **total_nav** — The total Net Asset Value of the sub-fund.
3. **share_classes** — For every share class listed:
   - name, currency, NAV, and number of outstanding shares at period end.
4. **income_expenses** — Every income and expense line item from the \
Statement of Operations (or equivalent). Use positive amounts for income, \
negative for expenses.
5. **reporting_period_start** and **reporting_period_end** — As ISO 8601 dates.
6. **source_pages** — For each field group you populate, list the printed page \
numbers where you found the data (e.g. {"total_nav": [5], "share_classes": [5, 6]}).
7. **missing_fields** — If any of the above cannot be determined from the \
provided text, list the field names here.

Extraction guidelines:
- Page numbers in headers/footers are *printed* page numbers — report those.
- If the document uses thousands separators (commas or dots), parse them correctly.
- Currency codes should be ISO 4217 (USD, EUR, GBP, CHF, JPY, etc.).
- Do NOT invent data. If a value is not present, mark it as missing.\
"""


class SubFundExtractor:
    """Extracts structured financial data from sub-fund pages via a single LLM call."""

    def __init__(self, page_reader: PageReader, model: str = "claude-sonnet-4-6"):
        self.page_reader = page_reader
        self.client = Anthropic()
        self.model = model

    def extract(self, entry: SubFundEntry) -> SubFundResult:
        """Extract all required variables for a single sub-fund."""
        pages = self._collect_pages(entry)
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
            "Extracting data for '%s' (pages %d–%d, %d pages of text)",
            entry.name,
            entry.start_page,
            entry.end_page,
            len(pages),
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

    def extract_all(self, entries: list[SubFundEntry]) -> list[SubFundResult]:
        """Extract data for all sub-funds sequentially."""
        results: list[SubFundResult] = []
        for i, entry in enumerate(entries, 1):
            logger.info("Processing sub-fund %d/%d: '%s'", i, len(entries), entry.name)
            results.append(self.extract(entry))
        return results

    def _collect_pages(self, entry: SubFundEntry) -> dict[int, str]:
        """Gather page text for the sub-fund, preferring targeted sections when available."""
        targeted: dict[int, str] = {}

        net_assets_section = self._find_section(entry, _NET_ASSETS_KEYWORDS)
        operations_section = self._find_section(entry, _OPERATIONS_KEYWORDS)

        if net_assets_section:
            targeted.update(self.page_reader.get_range_text(
                net_assets_section.start_page, net_assets_section.end_page,
            ))
        if operations_section:
            targeted.update(self.page_reader.get_range_text(
                operations_section.start_page, operations_section.end_page,
            ))

        if targeted:
            return targeted

        return self.page_reader.get_range_text(entry.start_page, entry.end_page)

    @staticmethod
    def _find_section(
        entry: SubFundEntry,
        keywords: frozenset[str],
    ) -> Any | None:
        """Find the first section whose title matches any of the keywords."""
        for section in entry.sections:
            title_lower = section.title.lower()
            if any(kw in title_lower for kw in keywords):
                return section
        return None

    @staticmethod
    def _build_user_message(subfund_name: str, pages: dict[int, str]) -> str:
        header = (
            f"Sub-fund: **{subfund_name}**\n\n"
            f"Below are the relevant pages from this sub-fund's section of the report.\n"
        )

        page_blocks = []
        for page_num in sorted(pages):
            text = pages[page_num]
            if text.strip():
                page_blocks.append(
                    f"--- PRINTED PAGE {page_num} ---\n{text}"
                )

        footer = (
            "Extract all available financial data for this sub-fund "
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
