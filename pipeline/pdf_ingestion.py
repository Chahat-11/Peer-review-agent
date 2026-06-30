import fitz  # PyMuPDF
import re

def extract_paper(pdf_path):
    """
    Reads a PDF and returns a dict with full_text + individual sections.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    sections = split_sections(full_text)

    return {
        "full_text": full_text,
        "abstract": sections.get("abstract", ""),
        "introduction": sections.get("introduction", ""),
        "methods": sections.get("methods", ""),
        "results": sections.get("results", ""),
        "conclusion": sections.get("conclusion", ""),
    }

def split_sections(text):
    """
    Detect section boundaries using keyword matching.
    Returns a dict: section_name -> section_text
    """
    section_patterns = [
        (r"abstract", "abstract"),
        (r"introduction", "introduction"),
        (r"(?:related work|literature review|background)", "related_work"),
        (r"(?:method|methodology|approach|proposed)", "methods"),
        (r"(?:experiment|evaluation|results|empirical)", "results"),
        (r"(?:discussion|analysis)", "discussion"),
        (r"(?:conclusion|conclusions|summary)", "conclusion"),
        (r"(?:reference|bibliography)", "references"),
    ]

    headers = []
    for pattern, name in section_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.start())
            line = text[line_start:line_end].strip()
            if len(line) < 60:
                headers.append((match.start(), name))

    headers.sort(key=lambda x: x[0])

    sections = {}
    for i, (pos, name) in enumerate(headers):
        if name == "references":
            break
        start = pos
        if i + 1 < len(headers):
            end = headers[i + 1][0]
        else:
            end = len(text)
        sections[name] = text[start:end].strip()

    return sections

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_ingestion.py <pdf_path>")
        sys.exit(1)

    result = extract_paper(sys.argv[1])
    print(f"Full text length: {len(result['full_text'])} chars")
    print(f"Abstract length:  {len(result['abstract'])} chars")
    print(f"Sections found:   {[k for k, v in result.items() if v]}")
    print()
    if result['abstract']:
        print("--- Abstract preview (first 200 chars) ---")
        print(result['abstract'][:200])