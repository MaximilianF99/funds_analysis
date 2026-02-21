import re
from pypdf import PdfReader
import logging

logging.getLogger("pypdf").setLevel(logging.ERROR)

def find_probable_toc_pages(pdf_path: str, top_n: int = 5) -> list[dict]:
    reader = PdfReader(pdf_path)
    page_scores = []
    
    toc_line_pattern = re.compile(r".+\s\.{2,}\s*\d+$|.+\s+\d+$")

    high_prob_keywords = [
        "table of contents", "table of content", 
        "inhaltsverzeichnis", "table des matières"
    ]
    medium_prob_keywords = [
        "contents", "content", "sommaire", "inhalt"
    ]
    low_prob_keywords = [
        "index"
    ]
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
            
        score = 0
        lines = text.split('\n')

        top_text = " ".join(lines[:15]).lower()

        if any(kw in top_text for kw in high_prob_keywords):
            score += 40
        elif any(kw in top_text for kw in medium_prob_keywords):
            score += 20
        elif any(kw in top_text for kw in low_prob_keywords):
            score += 5
            
        for line in lines:
            if toc_line_pattern.match(line.strip()):
                score += 1
                
        if score > 20:
            page_scores.append({
                "page_num": i + 1,
                "score": score,
                "text": text
            })
            
    page_scores.sort(key=lambda x: x["score"], reverse=True)
    
    best_pages = sorted(page_scores[:top_n], key=lambda x: x["page_num"])
    
    return best_pages

# === Testlauf ===
if __name__ == "__main__":
    pdf_path = "funds_files\Goldman_Sachs_Funds_PLC_SAR.pdf"
    toc_pages = find_probable_toc_pages(pdf_path)
    
    for p in toc_pages:
        print(f"Seite {p['page_num']} (Score: {p['score']})")
