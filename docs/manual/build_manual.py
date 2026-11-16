"""Builds the manuals from Markdown: the user manual (en.md, de.md), the
technical reference (technical_reference_en.md) and the developer guide
(docs/developer/*.md, joined into one document), each as

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

The developer guide stays readable on GitHub as separate pages with ordinary
relative links. Here each page is rendered on its own - headings one level
down, heading ids prefixed with the page's id, links between its pages turned
into links inside the one document, other relative links into plain text -
and the pages are joined in `DEVELOPER_PAGES` order.
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
from markdown.extensions.toc import slugify

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / "build"
HELP = BUILD / "help"
sys.path.insert(0, str(ROOT / "source"))

from bictool import API_VERSION, __version__  # noqa: E402
from bictool.infrastructure.app_paths import (  # noqa: E402
    DEVELOPER_GUIDE_FILE,
    HELP_INDEX_FILE,
    MANUAL_FILES,
    TECHNICAL_REFERENCE_FILE,
)

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
CONTENTS = {"en": "Contents", "de": "Inhalt"}
HELP_IMAGE_WIDTH = 640  # px; screenshots are scaled down to this in the help window


@dataclass(frozen=True)
class Document:
    key: str        # "user_guide" | "technical_reference" | "developer_guide"
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
DEVELOPER_GUIDE = Document("developer_guide", "en", "", "BICTool Developer Guide", DEVELOPER_GUIDE_FILE)
DEVELOPER = ROOT / "docs" / "developer"
# The developer guide's pages in reading order (all pages; a test checks that none is missing).
DEVELOPER_PAGES = (
    "README.md",
    "tutorials/first-analysis.md",
    "tutorials/first-plugin.md",
    "how-to/check-rois-before-measuring.md",
    "how-to/choose-settings.md",
    "how-to/run-many-studies.md",
    "how-to/save-and-read-results.md",
    "how-to/metric-with-own-roi.md",
    "how-to/artefact-hints.md",
    "how-to/wmgm-method.md",
    "how-to/export-format.md",
    "how-to/test-a-plugin.md",
    "how-to/install-and-share-a-plugin.md",
    "how-to/troubleshooting.md",
    "reference/api-functions.md",
    "reference/rois.md",
    "reference/settings.md",
    "reference/results.md",
    "reference/errors.md",
    "reference/json-format.md",
    "reference/command-line.md",
    "reference/hooks.md",
    "reference/plugin-types.md",
    "reference/changelog.md",
    "explanation/how-a-study-is-run.md",
    "explanation/plugin-system.md",
    "explanation/data-protection.md",
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
pre { background: #f1f4f6; padding: 5pt 7pt; white-space: pre-wrap; font-size: 9pt; break-inside: avoid; }
pre code { background: none; padding: 0; }
h4 { font-size: 11pt; margin: 14pt 0 3pt 0; padding-top: 5pt; border-top: 1px solid #dde4e8; break-after: avoid; }
h4 code { font-size: 11pt; background: none; padding: 0; }
.dev-page { margin-top: 26pt; }
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
pre { background-color: #f1f4f6; }
h4 { font-size: 11pt; margin-top: 10px; }
h4 code { font-size: 11pt; background-color: #ffffff; }
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


# ---- the developer guide (many pages, one document) ----

_FENCE = re.compile(r"^(```|~~~)")
_MD_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")


def _page_id(page: str) -> str:
    return "dev-" + re.sub(r"[^a-z0-9]+", "-", page.removesuffix(".md").lower()).strip("-")


def _outside_code(text: str, change) -> str:
    """`change(line)` for every line outside fenced code blocks."""
    out, fenced = [], False
    for line in text.split("\n"):
        if _FENCE.match(line.strip()):
            fenced = not fenced
            out.append(line)
        else:
            out.append(line if fenced else change(line))
    return "\n".join(out)


def _page_link(page: str, match: re.Match) -> str:
    label, target = match.groups()
    if re.match(r"[a-z][a-z0-9+.-]*:", target):  # http(s), mailto, ...
        return match.group(0)
    file, _, anchor = target.partition("#")
    destination = page if not file else (Path(page).parent / file).as_posix()
    parts: list[str] = []
    for part in destination.split("/"):
        if part == "..":
            parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    destination = "/".join(parts)
    if destination not in DEVELOPER_PAGES:
        return label  # e.g. a file of the repository: plain text in the built guide
    return f"[{label}](#{_page_id(destination)}{'-' + anchor if anchor else ''})"


def developer_guide() -> tuple[str, list[dict]]:
    """The developer guide's body (HTML) and its headings (toc tokens: page = level 2)."""
    bodies, tokens = [], []
    for page in DEVELOPER_PAGES:
        page_id = _page_id(page)
        text = (DEVELOPER / page).read_text(encoding="utf-8")

        def change(line: str, page=page) -> str:
            line = "#" + line if line.startswith("#") else line  # one level down: a page is a chapter
            return _MD_LINK.sub(lambda m: _page_link(page, m), line)

        converter = markdown.Markdown(
            extensions=["tables", "fenced_code", "toc"],
            extension_configs={"toc": {"slugify": lambda value, separator, prefix=page_id:
                                       f"{prefix}-{slugify(value, separator)}"}},
        )
        body = converter.convert(_outside_code(text, change))
        bodies.append(f'<section class="dev-page"><a id="{page_id}"></a>{body}</section>')
        tokens += converter.toc_tokens
    return "\n".join(bodies), tokens


def _toc_html(tokens: list[dict], title: str) -> str:
    def items(level_tokens) -> str:
        rows = []
        for token in level_tokens:
            children = [t for t in token["children"] if t["level"] <= 3]
            inner = f"<ul>{items(children)}</ul>" if children else ""
            rows.append(f'<li><a href="#{token["id"]}">{token["name"]}</a>{inner}</li>')
        return "".join(rows)

    return f'<div class="contents"><span class="toctitle">{title}</span><ul>{items(tokens)}</ul></div>'


def _developer_title() -> str:
    return (
        '<div class="title-page"><h1>BICTool Developer Guide</h1>'
        f'<p class="sub">The Python API and plug-ins<br>BICTool {__version__} · API {API_VERSION}</p>'
        '<p class="sub">Developed for the German Federal Office for Radiation Protection (BfS)</p>'
        '<div class="note"><p><strong>For developers and medical physicists who script BICTool or extend it.'
        "</strong> How to use the program is described in the <em>User Manual</em>, how BICTool computes its "
        "results in the <em>Technical Reference</em>. BICTool is decision support only; it is not a medical "
        "device.</p></div></div>"
    )


def developer_pdf_html() -> Path:
    body, tokens = developer_guide()
    page = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{DEVELOPER_GUIDE.title}</title>'
        f"<style>{CSS}</style></head><body>{_developer_title()}{_toc_html(tokens, CONTENTS['en'])}{body}"
        "</body></html>"
    )
    path = BUILD / "developer_guide_en_print.html"
    path.write_text(page, encoding="utf-8")
    return path


def developer_help(images: Path) -> dict:
    body, tokens = developer_guide()
    page = (
        f"<html><head><meta charset=\"utf-8\"><title>{DEVELOPER_GUIDE.title}</title><style>{HELP_CSS}</style>"
        f"</head><body>{_help_body(_developer_title() + body, images)}</body></html>"
    )
    (HELP / DEVELOPER_GUIDE.help_file).write_text(page, encoding="utf-8")
    return {
        "key": DEVELOPER_GUIDE.key,
        "language": DEVELOPER_GUIDE.language,
        "title": DEVELOPER_GUIDE.title,
        "file": DEVELOPER_GUIDE.help_file,
        "sections": _sections(body, tokens),
    }


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

    entries = [to_help(document, args.images) for document in DOCUMENTS] + [developer_help(args.images)]
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
    pdf = BUILD / DEVELOPER_GUIDE.pdf
    to_pdf(developer_pdf_html(), pdf)
    print(f"Built {pdf}")


if __name__ == "__main__":
    main()
