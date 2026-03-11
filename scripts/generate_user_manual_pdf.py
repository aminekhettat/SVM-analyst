"""Generate a PDF version of USER_MANUAL.md.

This script renders the markdown manual as plain wrapped text into a multipage PDF
without requiring external tools like pandoc.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT_MD = ROOT / "USER_MANUAL.md"
OUTPUT_PDF = ROOT / "docs" / "SVM-Analyst-User-Manual-1.0.1.pdf"


def _normalize_markdown(lines: list[str]) -> list[str]:
    normalized: list[str] = []
    in_code = False

    for raw in lines:
        line = raw.rstrip("\n")

        if line.strip().startswith("```"):
            in_code = not in_code
            continue

        if in_code:
            normalized.append(f"    {line}")
            continue

        if line.startswith("### "):
            normalized.append("")
            normalized.append(line.replace("### ", "", 1).upper())
            normalized.append("")
            continue

        if line.startswith("## "):
            normalized.append("")
            normalized.append(line.replace("## ", "", 1).upper())
            normalized.append("")
            continue

        if line.startswith("# "):
            normalized.append(line.replace("# ", "", 1).upper())
            normalized.append("")
            continue

        normalized.append(line)

    return normalized


def _wrap_lines(lines: list[str], width: int = 98) -> list[str]:
    wrapped: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            wrapped.append("")
            continue

        if line.startswith("- ") or line.startswith("1. "):
            indent = " " * (2 if line.startswith("- ") else 3)
            bullet = line[:2] if line.startswith("- ") else "1. "
            content = line[2:] if line.startswith("- ") else line[3:]
            chunks = textwrap.wrap(content, width=width - len(indent))
            if chunks:
                wrapped.append(f"{bullet}{chunks[0]}")
                wrapped.extend(f"{indent}{chunk}" for chunk in chunks[1:])
            else:
                wrapped.append(line)
            continue

        if line.startswith("    "):
            wrapped.extend(textwrap.wrap(line, width=width) or [""])
            continue

        wrapped.extend(textwrap.wrap(line, width=width) or [""])

    return wrapped


def generate_pdf(input_md: Path, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    lines = input_md.read_text(encoding="utf-8").splitlines()
    normalized = _normalize_markdown(lines)
    wrapped = _wrap_lines(normalized)

    lines_per_page = 52
    pages = [
        wrapped[i : i + lines_per_page] for i in range(0, len(wrapped), lines_per_page)
    ]

    with PdfPages(output_pdf) as pdf:
        for i, page_lines in enumerate(pages, start=1):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle("SVM Analyst User Manual", fontsize=16, y=0.98)
            fig.text(0.5, 0.02, f"Page {i}", ha="center", fontsize=8, color="gray")
            fig.text(
                0.06,
                0.94,
                "\n".join(page_lines),
                va="top",
                family="monospace",
                fontsize=9,
            )
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    generate_pdf(INPUT_MD, OUTPUT_PDF)
    print(f"Generated: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
