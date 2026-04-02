from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter


def build_excerpt(
    source_pdf: str | Path = "Barin_Extraction/Thermochemical Data of Pure Substances - 1995 - Barin.pdf",
    output_pdf: str | Path = "Barin_Extraction/extract.pdf",
    *,
    start_page_index: int = 920,
    page_count: int = 6,
) -> Path:
    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    for page_index in range(start_page_index, start_page_index + page_count):
        writer.add_page(reader.pages[page_index])
    output_path = Path(output_pdf)
    with output_path.open("wb") as fh:
        writer.write(fh)
    return output_path


if __name__ == "__main__":
    print(build_excerpt())
