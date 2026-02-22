from __future__ import annotations

from pydantic import BaseModel, Field


class TOCPage(BaseModel):
    """Raw page data from PDFNavigator's TOC detection algorithm."""

    page_number: int = Field(description="1-based PDF page number")
    confidence_score: int = Field(
        description="Heuristic score — higher values indicate stronger TOC signals"
    )
    raw_text: str = Field(description="Full extracted text of the page")

    @classmethod
    def from_navigator_result(cls, raw: dict) -> TOCPage:
        """Convert raw dict output from find_probable_toc_pages() into a typed model."""
        return cls(
            page_number=raw["page_num"],
            confidence_score=raw["score"],
            raw_text=raw["text"],
        )


class TOCSection(BaseModel):
    """Named section within a sub-fund (e.g., 'Statement of Net Assets')."""

    title: str = Field(description="Section heading as printed in the TOC")
    start_page: int = Field(
        description="Page number where this section starts (printed page number)"
    )
    end_page: int = Field(
        description=(
            "Last page of this section (printed page number). "
            "This is the page immediately before the next TOC entry starts, "
            "whether that is another section or the next sub-fund."
        )
    )


class SubFundEntry(BaseModel):
    """Sub-fund entry extracted from the Table of Contents."""

    name: str = Field(description="Official sub-fund name as listed in the TOC")
    start_page: int = Field(
        description="Page number where this sub-fund's content begins (printed page number)"
    )
    end_page: int = Field(
        description=(
            "Last page belonging to this sub-fund (printed page number). "
            "This is the page immediately before the next TOC entry starts, "
            "whether that entry is another sub-fund or a general section."
        )
    )
    sections: list[TOCSection] = Field(
        default_factory=list,
        description="Named sections within this sub-fund, if the TOC provides that granularity",
    )


class ParsedTOC(BaseModel):
    """Structured Table of Contents of a fund report."""

    master_fund_name: str = Field(description="Name of the umbrella / master fund")
    subfunds: list[SubFundEntry] = Field(
        description="All sub-funds listed in the TOC, ordered by start_page"
    )


# ---------------------------------------------------------------------------
# Extraction output models
# ---------------------------------------------------------------------------


class ShareClassData(BaseModel):
    """Financial data for a single share class within a sub-fund."""

    name: str = Field(description="Share class name (e.g., 'Class A', 'Class I USD')")
    currency: str = Field(description="Currency of this share class (ISO 4217, e.g. 'USD')")
    nav: float | None = Field(
        default=None,
        description="Total Net Asset Value for this share class",
    )
    outstanding_shares: float | None = Field(
        default=None,
        description="Number of outstanding shares at the end of the reporting period",
    )


class IncomeExpenseItem(BaseModel):
    """Single line item from the Statement of Operations."""

    name: str = Field(description="Name of the income or expense position as printed")
    amount: float = Field(
        description="Amount in fund currency (positive = income, negative = expense)"
    )


class SubFundResult(BaseModel):
    """Extracted financial data for a single sub-fund."""

    subfund_name: str = Field(description="Official sub-fund name")
    fund_currency: str | None = Field(
        default=None,
        description="Base currency of the sub-fund (ISO 4217)",
    )
    total_nav: float | None = Field(
        default=None,
        description="Total Net Asset Value of the sub-fund",
    )
    share_classes: list[ShareClassData] = Field(
        default_factory=list,
        description="Per-share-class breakdown (NAV, currency, outstanding shares)",
    )
    income_expenses: list[IncomeExpenseItem] = Field(
        default_factory=list,
        description="All income and expense line items from the Statement of Operations",
    )
    reporting_period_start: str | None = Field(
        default=None,
        description="Start of the reporting period (ISO 8601, e.g. '2023-01-01')",
    )
    reporting_period_end: str | None = Field(
        default=None,
        description="End of the reporting period (ISO 8601, e.g. '2023-12-31')",
    )
    source_pages: dict[str, list[int]] = Field(
        default_factory=dict,
        description=(
            "Maps field group to the printed page numbers where the data was found "
            "(e.g. {'total_nav': [5], 'income_expenses': [8, 9]})"
        ),
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Names of fields that could not be extracted from the provided pages",
    )


class ExtractionReport(BaseModel):
    """Top-level output of the full extraction pipeline."""

    source_file: str = Field(description="Name of the source PDF file")
    master_fund_name: str = Field(description="Name of the umbrella / master fund")
    subfunds: list[SubFundResult] = Field(description="Extraction results per sub-fund")
