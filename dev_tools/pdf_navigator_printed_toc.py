import re
from typing import Dict, Tuple
from pypdf import PdfReader

class PDFNavigator:
    def __init__(self, pdf_path: str):
        self.reader = PdfReader(pdf_path)
        self.page_count = len(self.reader.pages)

    def get_subfund_page_ranges(self) -> Dict[str, Tuple[int, int]]:
        """
        Extracts subfund page ranges using PDF outlines or fallback regex.
        
        Returns:
            Dict[str, Tuple[int, int]]: Mapping of section titles to (start_page, end_page).
        """
        outline = self.reader.outline
        
        if not outline:
            return self._fallback_parse_printed_toc()

        cleaned_toc = []
        for item in outline:
            if isinstance(item, list):
                continue
            title = item.title.strip()
            page_num = self.reader.get_destination_page_number(item) + 1
            cleaned_toc.append({"title": title, "start_page": page_num})
            
        if len(cleaned_toc) < 10:
            return self._fallback_parse_printed_toc()

        return self._build_ranges(cleaned_toc)

    def _fallback_parse_printed_toc(self) -> Dict[str, Tuple[int, int]]:
        pattern = re.compile(r"([A-Za-z0-9\s\-\&]+?)\s*\.{3,}\s*(\d+)")
        found_funds = []
        
        for page_num in range(min(20, self.page_count)):
            text = self.reader.pages[page_num].extract_text()
            if not text:
                continue
                
            matches = pattern.findall(text)
            for title, page_str in matches:
                if len(title) > 3:
                    found_funds.append({"title": title.strip(), "start_page": int(page_str)})
                    
        if not found_funds:
            raise ValueError("No TOC found.")
            
        return self._build_ranges(found_funds)

    def _build_ranges(self, items: list[dict]) -> Dict[str, Tuple[int, int]]:
        page_ranges = {}
        for i, current in enumerate(items):
            start_page = current["start_page"]
            end_page = items[i + 1]["start_page"] - 1 if i + 1 < len(items) else self.page_count
            page_ranges[current["title"]] = (start_page, max(start_page, end_page))
        return page_ranges


if __name__ == "__main__":
    pdf_path = "funds_files/Allianz Global Investors Fund.pdf" 
    try:
        navigator = PDFNavigator(pdf_path)
        ranges = navigator.get_subfund_page_ranges()
        
        for fund, (start, end) in list(ranges.items())[:5]:
            print(f"{fund}: {start} - {end}")
    except Exception as e:
        print(f"Error: {e}")