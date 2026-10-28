"""Builds the manuals from Markdown: the user manual (en.md, de.md) and the
technical reference (technical_reference_en.md), each as

* a PDF (print stylesheet, A4, via Microsoft Edge in headless mode), and
* an HTML page for BICTool's help window, plus `help_index.json` (contents
  and full-text search; read by `integration/help_files.py`).

    .venv/Scripts/python.exe docs/manual/build_manual.py [--images FOLDER]

Output: docs/manual/build/ (not in git), picked up by
packaging/build_portable.py. The screenshots come from a real, anonymised
study and are not in git either (default folder: debugs/manual_shots).

Links between the documents are written `[text](tr:anchor)` (technical
reference) or `[text](ug:anchor)` (user manual, same language): in the help
pages they become links, in the PDFs plain text.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / "build"
HELP = BUILD / "help"
sys.path.insert(0, str(ROOT / "source"))

from bictool import __version__  # noqa: E402
from bictool.infrastructure.app_paths import HELP_INDEX_FILE, MANUAL_FILES, TECHNICAL_REFERENCE_FILE  # noqa: E402

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
CONTENTS = {"en": "Contents", "de": "Inhalt"}
HELP_IMAGE_WIDTH = 640  # px; screenshots are scaled down to this in the help window


@dataclass(frozen=True)
class Document:
    key: str        # "user_guide" | "technical_reference"
    language: str
    source: str     # Markdown file in this folder
    title: str
    pdf: str

    @property
    def help_file(self) -> str:
        return f"{self.key}_{self.language}.html"


DOCUMENTS = (
    Document("user_guide", "en", "en.md", "BICTool User Manual", MANUAL_FILES["en"]),
    Document("user_guide", "de", "de.md", "BICTool Benutzerhandbuch", MANUAL_FILES["de"]),
    Document("technical_reference", "en", "technical_reference_en.md", "BICTool Technical Reference",
             TECHNICAL_REFERENCE_FILE),
)

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1d2329; }
h1 { font-size: 22pt; margin: 0 0 4pt 0; color: #0f5560; }
h2 { font-size: 15pt; margin: 22pt 0 6pt 0; color: #0f5560; border-bottom: 1px solid #c9d3d8; padding-bottom: 2pt;
     break-after: avoid; }
h3 { font-size: 12pt; margin: 14pt 0 4pt 0; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
img { max-width: 100%; border: 1px solid #c9d3d8; margin: 4pt 0 2pt 0; }
img.image { width: 62%; } img.box { width: 58%; }
figure { margin: 6pt 0 10pt 0; break-inside: avoid; }
figcaption { font-size: 9pt; color: #5b6670; }
table { border-collapse: collapse; margin: 6pt 0; width: 100%; break-inside: avoid; }
th, td { border: 1px solid #c9d3d8; padding: 3pt 6pt; text-align: left; vertical-align: top; }
th { background: #eef3f5; }
code { font-family: Consolas, monospace; font-size: 9.5pt; background: #f1f4f6; padding: 0 2pt; }
blockquote { border-left: 3px solid #0f5560; margin: 8pt 0; padding: 2pt 10pt; background: #eef3f5; }
.note { border-left: 3px solid #d9a400; background: #fff8e1; padding: 5pt 8pt; margin: 8pt 0; }
.title-page { height: 200mm; display: flex; flex-direction: column; justify-content: center; }
.contents { break-before: page; break-after: page; }
.toctitle { display: block; font-weight: bold; margin-bottom: 6pt; }
.title-page .sub { font-size: 13pt; color: #5b6670; }
.contents ul { list-style: none; padding-left: 0; } .contents ul ul { padding-left: 14pt; }
"""

# Qt's rich-text engine (QTextBrowser) knows only a subset of CSS: no flex,
# no max-width, borders per table rather than per cell.
HELP_CSS = """
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; color: #1d2329; }
h1 { font-size: 18pt; color: #0f5560; }
h2 { font-size: 14pt; color: #0f5560; margin-top: 18px; }
h3 { font-size: 11.5pt; margin-top: 12px; }
th { background-color: #eef3f5; }
code { font-family: Consolas, monospace; background-color: #f1f4f6; }
.note { background-color: #fff8e1; }
blockquote { background-color: #eef3f5; }
figcaption, .caption { color: #5b6670; font-size: 9pt; }
.sub { color: #5b6670; }
a { color: #0f6b78; }
"""

_CROSS_LINK = re.compile(r"\[([^\]]+)\]\((tr|ug):([\w-]+)\)")
_CONTENTS_BLOCK = re.compile(r'<div class="contents" markdown="1">.*?</div>', re.S)


def _source_text(document: Document) -> str:
    return (HERE / document.source).read_text(encoding="utf-8").replace("{{VERSION}}", __version__)


def _converter(document: Document) -> markdown.Markdown:
    return markdown.Markdown(
        extensions=["tables", "attr_list", "md_in_html", "toc"],
        extension_configs={"toc": {"toc_depth": "2-3", "title": CONTENTS[document.language]}},
    )


# ---- PDF ----

def to_pdf_html(document: Document) -> Path:
    text = _CROSS_LINK.sub(r"*\1*", _source_text(document))
    body = _converter(document).convert(text)
    page = (
        f"<!doctype html><html lang=\"{document.language}\"><head><meta charset=\"utf-8\">"
        f"<title>{document.title}</title><style>{CSS}</style></head><body>{body}</body></html>"
    )
    path = BUILD / f"{document.key}_{document.language}_print.html"
    path.write_text(page, encoding="utf-8")
    return path


def to_pdf(html: Path, pdf: Path) -> None:
    """Own profile per call (otherwise Edge may hand the job to a running instance), and wait for the file:
    Edge can return before it has written the PDF."""
    pdf.unlink(missing_ok=True)
    profile = tempfile.mkdtemp(prefix="bictool_edge_")
    subprocess.run(
        [str(EDGE), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
         f"--print-to-pdf={pdf}", html.as_uri()],
        check=True, capture_output=True, timeout=180,
    )
    size, deadline = -1, time.time() + 120
    while time.time() < deadline:
        current = pdf.stat().st_size if pdf.is_file() else -1
        if current > 0 and current == size:
            break
        size = current
        time.sleep(1)
    else:
        raise RuntimeError(f"Edge did not write {pdf}")
    shutil.rmtree(profile, ignore_errors=True)


# ---- help pages ----

def _png_width(path: Path) -> int | None:
    with path.open("rb") as file:
        header = file.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">I", header[16:20])[0]


def _help_body(body: str, images: Path) -> str:
    """Adapts the HTML to Qt's rich-text engine: tables get border attributes,
    screenshots a fixed width, notes and figures a frame."""
    body = body.replace("<table>", '<table border="1" cellspacing="0" cellpadding="5" width="100%">')

    def image(match: re.Match) -> str:
        tag, source = match.group(0), match.group(1)
        width = _png_width(images / Path(source).name) if (images / Path(source).name).is_file() else None
        if width is None:
            return tag
        return tag.replace("<img ", f'<img width="{min(width, HELP_IMAGE_WIDTH)}" ', 1)

    body = re.sub(r'<img [^>]*src="([^"]+)"[^>]*>', image, body)
    body = re.sub(r'<div class="note">(.*?)</div>',
                  r'<table class="note" width="100%" cellpadding="8"><tr><td>\1</td></tr></table>', body, flags=re.S)
    body = re.sub(r"<figcaption>(.*?)</figcaption>", r'<p class="caption">\1</p>', body, flags=re.S)
    return body


def _sections(body: str, toc_tokens: list[dict]) -> list[dict]:
    """Title, level and plain text of every chapter/section (h2/h3), for the contents tree and the search."""
    flat: list[dict] = []

    def walk(tokens):
        for token in tokens:
            if token["level"] in (2, 3):
                flat.append({"id": token["id"], "title": html_lib.unescape(token["name"]), "level": token["level"]})
            walk(token["children"])

    walk(toc_tokens)
    starts = []
    for section in flat:
        match = re.search(rf'<h{section["level"]} id="{re.escape(section["id"])}"', body)
        if match is None:
            raise ValueError(f"Heading {section['id']} not found")
        starts.append(match.start())
    for k, section in enumerate(flat):
        end = starts[k + 1] if k + 1 < len(flat) else len(body)
        text = re.sub(r"<h[23][^>]*>.*?</h[23]>", " ", body[starts[k]:end], count=1, flags=re.S)  # title apart
        text = re.sub(r"<[^>]+>", " ", text)
        section["text"] = re.sub(r"\s+", " ", html_lib.unescape(text)).strip()
    return flat


def to_help(document: Document, images: Path) -> dict:
    """The help page of `document` and its entry in the help index."""
    language = document.language

    def cross(match: re.Match) -> str:
        text, kind, anchor = match.groups()
        target = (f"technical_reference_en.html#{anchor}" if kind == "tr"
                  else f"user_guide_{language}.html#{anchor}")
        return f"[{text}]({target})"

    text = _CONTENTS_BLOCK.sub("", _CROSS_LINK.sub(cross, _source_text(document)))
    converter = _converter(document)
    body = converter.convert(text)
    sections = _sections(body, converter.toc_tokens)
    page = (
        f"<html><head><meta charset=\"utf-8\"><title>{document.title}</title><style>{HELP_CSS}</style></head>"
        f"<body>{_help_body(body, images)}</body></html>"
    )
    (HELP / document.help_file).write_text(page, encoding="utf-8")
    return {
        "key": document.key,
        "language": language,
        "title": document.title,
        "file": document.help_file,
        "sections": sections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, default=ROOT / "debugs" / "manual_shots")
    parser.add_argument("--skip-pdf", action="store_true", help="build only the help pages")
    args = parser.parse_args()

    BUILD.mkdir(exist_ok=True)
    for folder in (BUILD / "images", HELP):
        if folder.exists():
            shutil.rmtree(folder)
    shutil.copytree(args.images, BUILD / "images")
    shutil.copytree(args.images, HELP / "images")

    entries = [to_help(document, args.images) for document in DOCUMENTS]
    (HELP / HELP_INDEX_FILE).write_text(
        json.dumps({"version": __version__, "documents": entries}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"Built help pages in {HELP}")
    if args.skip_pdf:
        return
    for document in DOCUMENTS:
        pdf = BUILD / document.pdf
        to_pdf(to_pdf_html(document), pdf)
        print(f"Built {pdf}")


if __name__ == "__main__":
    main()
