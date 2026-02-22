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
