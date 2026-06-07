#!/usr/bin/env python3
"""
Convierte docs/informe_tecnico.md a PDF de una carilla, sin pandoc ni LaTeX.

Requisitos (en el venv):
    pip install markdown weasyprint

Uso:
    python docs/md_to_pdf.py
    # genera docs/informe_tecnico.pdf
"""

from pathlib import Path

import markdown
from weasyprint import HTML

AQUI = Path(__file__).resolve().parent
MD_IN = AQUI / "informe_tecnico.md"
PDF_OUT = AQUI / "informe_tecnico.pdf"

# CSS tuneado para que el informe entre en una sola página A4.
CSS = """
@page { size: A4; margin: 1.4cm; }
* { box-sizing: border-box; }
body {
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 9.7pt;
    line-height: 1.34;
    color: #1a1a1a;
}
h1 { font-size: 19pt; margin: 0 0 1px 0; color: #0b3d63; }
h1 + h3 { font-size: 10.5pt; font-weight: 600; color: #2171b5;
          margin: 0 0 6px 0; font-style: italic; }
h2 { font-size: 11.5pt; color: #0b3d63; margin: 9px 0 3px 0;
     border-bottom: 1.5px solid #d7e3ee; padding-bottom: 1px; }
p { margin: 3px 0; }
ul { margin: 3px 0; padding-left: 17px; }
li { margin: 1px 0; }
strong { color: #0b3d63; }
hr { border: none; border-top: 1px solid #d7e3ee; margin: 6px 0; }
code, pre {
    font-family: "DejaVu Sans Mono", monospace;
    font-size: 8.6pt; background: #f4f7fa;
}
pre { padding: 5px 8px; border-radius: 4px; margin: 4px 0;
      border: 1px solid #e3ebf2; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin: 4px 0; font-size: 9.2pt; }
th, td { border: 1px solid #cdd9e5; padding: 2.5px 7px; text-align: left; }
th { background: #eaf1f8; color: #0b3d63; }
sub { color: #555; font-size: 8pt; }
"""


def main():
    md_text = MD_IN.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists", "fenced_code"])
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>" \
           f"<style>{CSS}</style></head><body>{body}</body></html>"
    HTML(string=html).write_pdf(str(PDF_OUT))
    print(f"PDF generado: {PDF_OUT}")


if __name__ == "__main__":
    main()
