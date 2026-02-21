import fitz
from typing import Dict, Tuple

class PDFNavigator:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def get_subfund_page_ranges(self) -> Dict[str, Tuple[int, int]]:
        """
        Extracts the Table of Contents (TOC) from the PDF and calculates
        the start and end page for each section or subfund.
        
        Returns:
            Dict[str, Tuple[int, int]]: Dictionary mapping section titles to (start_page, end_page).
        """
        toc = self.doc.get_toc()
        if not toc:
            raise ValueError("PDF contains no bookmarks. Fallback parser required.")

        page_ranges = {}
        cleaned_toc = [{"title": item[1].strip(), "start_page": item[2]} for item in toc]

        for i, current in enumerate(cleaned_toc):
            title = current["title"]
            start_page = current["start_page"]
            
            if i + 1 < len(cleaned_toc):
                end_page = cleaned_toc[i + 1]["start_page"] - 1
            else:
                end_page = self.doc.page_count
                    
            if end_page < start_page:
                end_page = start_page
                
            page_ranges[title] = (start_page, end_page)

        return page_ranges


if __name__ == "__main__":
    pdf_path = "funds_files\HSBC Global Investment Fund_FYE 31.03.2023.pdf"
    
    try:
        navigator = PDFNavigator(pdf_path)
        ranges = navigator.get_subfund_page_ranges()
        
        search_fund = "Allianz Strategic Bond"
        
        for fund_name, (start, end) in ranges.items():
            if search_fund.lower() in fund_name.lower():
                print(f"Found: '{fund_name}'")
                print(f"Page Range: {start} to {end}")
                break
                
    except Exception as e:
        print(f"Error: {e}")