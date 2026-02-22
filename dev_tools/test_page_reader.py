import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from page_reader import PageReader
from models import SubFundEntry, ParsedTOC

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)

toc = ParsedTOC(
    master_fund_name="Goldman Sachs Funds, plc",
    subfunds=[
        SubFundEntry(name="Goldman Sachs US$ Liquid Reserves Fund", start_page=5, end_page=9, sections=[]),
        SubFundEntry(name="Goldman Sachs Euro Liquid Reserves Fund", start_page=14, end_page=17, sections=[]),
        SubFundEntry(name="Goldman Sachs US$ Standard VNAV Fund", start_page=25, end_page=29, sections=[]),
    ],
)

reader = PageReader("funds_files/AGI.pdf")
reader.calibrate(toc)

print(f"\nOffset: {reader.offset}")
print(f"Printed page 5 → PDF index {reader.printed_to_index(5)}\n")
print("--- Page 5 (first 500 chars) ---")
print(reader.get_page_text(1))
