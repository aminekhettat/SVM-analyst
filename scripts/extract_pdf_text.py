"""Utility to extract text from the thesis PDF for analysis.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

from PyPDF2 import PdfReader
import sys


if __name__ == "__main__":
    reader = PdfReader("docs/256161.pdf")
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end = int(sys.argv[2]) if len(sys.argv) > 2 else min(3, len(reader.pages))
    print("Pages:", len(reader.pages))
    for i in range(start - 1, min(end, len(reader.pages))):
        print("--- page", i + 1, "---")
        text = reader.pages[i].extract_text() or ""
        print(text[:5000])
        print("\n")
