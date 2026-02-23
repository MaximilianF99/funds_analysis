"""Unit tests for pure helper functions and Pydantic model validators."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import ShareClassData, SubFundEntry, TOCPage, TOCSection
from subfund_extractor import SubFundExtractor, _normalize_ws
from llm_client import resolve_refs
from page_reader import PageReader


# ---------------------------------------------------------------------------
# _normalize_ws
# ---------------------------------------------------------------------------

class TestNormalizeWs:
    def test_collapses_multiple_spaces(self):
        assert _normalize_ws("hello   world") == "hello world"

    def test_collapses_tabs_and_newlines(self):
        assert _normalize_ws("hello\t\n  world") == "hello world"

    def test_strips_leading_and_trailing(self):
        assert _normalize_ws("  hello  ") == "hello"

    def test_empty_string(self):
        assert _normalize_ws("") == ""


# ---------------------------------------------------------------------------
# resolve_refs  (JSON Schema $ref inlining)
# ---------------------------------------------------------------------------

class TestResolveRefs:
    def test_inlines_simple_ref(self):
        schema = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }
            },
            "type": "object",
            "properties": {
                "address": {"$ref": "#/$defs/Address"},
            },
        }
        result = resolve_refs(schema)
        assert "$defs" not in result
        assert result["properties"]["address"] == {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        }

    def test_no_defs_returns_schema_unchanged(self):
        schema = {"type": "string"}
        assert resolve_refs(schema) == {"type": "string"}

    def test_nested_refs(self):
        schema = {
            "$defs": {
                "Inner": {"type": "integer"},
                "Outer": {
                    "type": "object",
                    "properties": {"value": {"$ref": "#/$defs/Inner"}},
                },
            },
            "type": "object",
            "properties": {"data": {"$ref": "#/$defs/Outer"}},
        }
        result = resolve_refs(schema)
        assert result["properties"]["data"]["properties"]["value"] == {"type": "integer"}


# ---------------------------------------------------------------------------
# ShareClassData  (NAV auto-computation validator)
# ---------------------------------------------------------------------------

class TestShareClassDataValidator:
    def test_computes_nav_from_per_share_and_outstanding(self):
        sc = ShareClassData(
            name="Class A",
            currency="EUR",
            nav_per_share=120.50,
            outstanding_shares=10_000,
        )
        assert sc.nav == round(120.50 * 10_000, 2)

    def test_does_not_overwrite_explicit_nav(self):
        sc = ShareClassData(
            name="Class B",
            currency="USD",
            nav=999_999.0,
            nav_per_share=100.0,
            outstanding_shares=5_000,
        )
        assert sc.nav == 999_999.0

    def test_nav_stays_none_when_data_missing(self):
        sc = ShareClassData(name="Class C", currency="CHF", nav_per_share=50.0)
        assert sc.nav is None


# ---------------------------------------------------------------------------
# PageReader._select_samples
# ---------------------------------------------------------------------------

class TestSelectSamples:
    @staticmethod
    def _make_entries(n: int) -> list[SubFundEntry]:
        return [
            SubFundEntry(name=f"Fund {i}", start_page=i * 10, end_page=i * 10 + 9)
            for i in range(n)
        ]

    def test_returns_all_when_fewer_than_num_samples(self):
        entries = self._make_entries(2)
        result = PageReader._select_samples(entries)
        assert len(result) == 2

    def test_returns_num_samples_when_many_entries(self):
        entries = self._make_entries(20)
        result = PageReader._select_samples(entries)
        assert len(result) == 3
        assert result[0] is entries[0]
        assert result[-1] is entries[-1]

    def test_single_entry(self):
        entries = self._make_entries(1)
        result = PageReader._select_samples(entries)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# SubFundExtractor._is_relevant_shared_section
# ---------------------------------------------------------------------------

class TestIsRelevantSharedSection:
    def test_matches_nav_keyword(self):
        assert SubFundExtractor._is_relevant_shared_section("Statement of Net Assets") is True

    def test_matches_income_keyword(self):
        assert SubFundExtractor._is_relevant_shared_section(
            "Statement of Operations and Changes"
        ) is True

    def test_rejects_unrelated_section(self):
        assert SubFundExtractor._is_relevant_shared_section("Auditor's Report") is False

    def test_case_insensitive(self):
        assert SubFundExtractor._is_relevant_shared_section("NOTES TO THE FINANCIAL STATEMENTS") is True
