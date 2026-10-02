"""
doc_tool.py - attach a document, humanize it, save a new PDF.

Stdlib-only PDF *writer*: builds a text PDF using the base-14 fonts, so
exporting needs no extra install. Reading PDF/DOCX is optional and detected
at runtime (pypdf / python-docx) - the app still runs without them.

Flow used by the GUI:
    doc = load_document(path)            # -> Document (pages of blocks)
    pages = humanize_document(engine, doc, intensity, on_progress)
    write_pdf(out_path, pages, doc.sizes)
"""
import os
import re
import zlib
from typing import Callable, List, Optional, Sequence, Tuple


# ----------------------------------------------------------------- optional
def pdf_reader():
    """(module, friendly_name) for the installed PDF reader, or None."""
    try:
        import pypdf
        return pypdf, "pypdf"
    except Exception:
        pass
    try:
        import PyPDF2 as legacy
        return legacy, "PyPDF2"
    except Exception:
        return None


def docx_reader():
    """python-docx module if installed, else None."""
    try:
        import docx
        return docx
    except Exception:
        return None


def readable_types() -> List[Tuple[str, str]]:
    """File-dialog filters for what can actually be opened right now."""
    out = []
    if pdf_reader():
        out.append(("PDF documents", "*.pdf"))
    if docx_reader():
        out.append(("Word documents", "*.docx"))
    out.append(("Text and Markdown", "*.txt *.md *.markdown"))
    return out


def missing_hint() -> str:
    """pip line for whichever readers are absent (empty when all present)."""
    need = []
    if not pdf_reader():
        need.append("pypdf")
    if not docx_reader():
        need.append("python-docx")
    return "pip install " + " ".join(need) if need else ""


# --------------------------------------------------------------- text tools
_ITEM_MARKERS = "\u2022\u2023\u25aa\u25cf\u2043\u2219\u00b7\u25e6\u2044" \
                "-*\u2013\u2014"
_ITEM_RE = re.compile(r"^[" + re.escape(_ITEM_MARKERS) + r"]\s+")
_ENUM_RE = re.compile(r"^(?:[a-zA-Z0-9]{1,3}[.)]\s+|o\s+)(?=\S)")
_NUM_HEAD_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?|[IVXLC]+\.|chapter\s+\d+|part\s+[ivxlc\d]+"
    r"|appendix\s+[a-z0-9]+)\s+\S", re.IGNORECASE)
_TERMINAL = ".?!,;:\"\u201d)"
_SENT_END = ".?!\"'\u201d"          # a wrapped line rarely stops on a comma
_HEAD_MAX_CHARS = 70
_HEAD_MAX_WORDS = 10
_BLOCK_MAX_WORDS = 90        # a soft-wrapped run never grows past this


def _is_blank(block) -> bool:
    return bool(block) and block.get("kind") == "blank"


def _item_text(stripped: str):
    """Item text if the line opens with a bullet marker, else None."""
    if not stripped or stripped[0] not in _ITEM_MARKERS:
        return None
    if not _ITEM_RE.match(stripped):
        return None
    rest = _ENUM_RE.sub("", _ITEM_RE.sub("", stripped, count=1)).strip()
    return rest or None


def _split_blocks(text: str) -> List[dict]:
    """Extracted page text -> flat list of layout blocks.

    kind: blank | para | head | bullet. PDF extraction keeps soft wraps as
    plain newlines, so wrapped lines are folded back into the block above.
    Headings and list items are "locked": the line after one starts a new
    block, so a title never swallows the paragraph under it and a bullet
    never swallows the paragraph after the list.
    """
    lines = [ln.rstrip() for ln in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    blocks: List[dict] = []
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            if blocks and not _is_blank(blocks[-1]):
                blocks.append({"kind": "blank", "text": ""})
            continue

        prev = blocks[-1] if blocks else None
        prev_blank = prev is None or _is_blank(prev)
        prev_done = (not prev_blank
                     and prev["text"].rstrip()[-1:] in _SENT_END)
        prev_locked = (not prev_blank
                       and (prev.get("locked") or prev["kind"] == "bullet"))

        # soft-wrapped continuation of the block above
        if not prev_blank and not prev_done and not prev_locked:
            merged = prev["text"] + " " + stripped
            if len(merged.split()) <= _BLOCK_MAX_WORDS:
                prev["text"] = merged
                if prev["kind"] == "head" and (len(merged) > _HEAD_MAX_CHARS
                                               or len(merged.split()) > _HEAD_MAX_WORDS):
                    prev["kind"] = "para"
                continue

        item = _item_text(stripped)
        if item and (prev_blank or prev_done or prev_locked):
            blocks.append({"kind": "bullet", "text": item, "locked": True})
            continue

        if _is_headish(stripped, lines, i):
            blocks.append({"kind": "head", "text": stripped, "locked": True})
        else:
            blocks.append({"kind": "para", "text": stripped})
    return blocks


def _is_headish(stripped: str, lines: Sequence[str], i: int) -> bool:
    """Short standalone line with no closing punctuation.

    Two tells only: numbered sections ("2.1 Risks", "Chapter 4"), or a
    title-cased line whose words are mostly capitalised - which a wrapped
    body sentence rarely is.
    """
    if stripped[-1:] in _TERMINAL:
        return False
    words = stripped.split()
    if len(stripped) > _HEAD_MAX_CHARS or len(words) > _HEAD_MAX_WORDS:
        return False
    if _NUM_HEAD_RE.match(stripped):
        return True
    nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
    if nxt and not nxt[:1].isupper():
        return False
    caps = sum(1 for w in words if w[:1].isupper())
    return caps >= 2 and caps >= 0.6 * len(words)


def _clean_item(item: str) -> str:
    return _ENUM_RE.sub("", _ITEM_RE.sub("", item, count=1)).strip()


def _join_blocks(blocks: Sequence[dict]) -> str:
    """Render blocks back to text the engine understands (paragraphs kept)."""
    texts = [b["text"].strip() for b in blocks
             if not _is_blank(b) and b["text"].strip()]
    return "\n\n".join(texts)


def text_of(pages: Sequence[Sequence[dict]]) -> str:
    """Render humanized pages back to plain text, one blank line per block."""
    return "\n\n".join(t for t in (_join_blocks(page) for page in pages) if t)


# ------------------------------------------------------------------ document
class Document:
    """An attached document: per-page blocks plus the original page sizes."""

    def __init__(self, path, kind, pages, sizes=None):
        self.path = path
        self.name = os.path.basename(path)
        self.kind = kind                      # pdf | docx | text
        self.pages = pages                    # List[List[block]]
        self.sizes = sizes or []

    @property
    def size_hint(self) -> str:
        if self.kind == "pdf":
            pages = len(self.pages)
            return f"{pages} page{'s' if pages != 1 else ''}"
        return f"{self.word_count:,} words"

    @property
    def word_count(self) -> int:
        return sum(len(b["text"].split()) for page in self.pages for b in page)

    def text(self) -> str:
        """Whole document as plain text, page breaks kept as blank lines."""
        return text_of(self.pages)


# --------------------------------------------------------------------- load
class DocError(Exception):
    pass


def load_document(path: str) -> Document:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    if ext in (".txt", ".md", ".markdown"):
        return _load_text(path)
    raise DocError(f"Unsupported file type: {ext or 'unknown'}")


def _load_pdf(path: str) -> Document:
    reader, name = pdf_reader()
    if reader is None:
        raise DocError(f"Reading PDFs needs {name or 'pypdf'}.\n\n"
                       "    pip install pypdf")
    try:
        with open(path, "rb") as fh:
            pdf = reader.PdfReader(fh)
            pages, sizes = [], []
            encrypted = getattr(pdf, "is_encrypted", False)
            if encrypted:
                try:
                    pdf.decrypt("")
                except Exception:
                    raise DocError("This PDF is password protected. "
                                   "Remove the password, then attach it again.")
            for page in pdf.pages:
                pages.append(_split_blocks(page.extract_text() or ""))
                box = getattr(page, "mediabox", None)
                sizes.append((float(box.width), float(box.height)) if box else None)
    except DocError:
        raise
    except Exception as e:
        raise DocError(f"Could not read the PDF: {e}") from e
    if not any(b["text"].strip() for page in pages for b in page):
        raise DocError("No selectable text in this PDF.\n\n"
                       "It looks like a scan - run it through OCR first, "
                       "then attach it again.")
    return Document(path, "pdf", pages, sizes)


def _load_docx(path: str) -> Document:
    docx = docx_reader()
    if docx is None:
        raise DocError("Reading Word files needs python-docx.\n\n"
                       "    pip install python-docx")
    try:
        document = docx.Document(path)
    except Exception as e:
        raise DocError(f"Could not read the Word file: {e}") from e

    blocks: List[dict] = []
    for para in document.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style is not None else ""
        if style.startswith("heading") or style in ("title", "subtitle"):
            kind = "head"
        elif style.startswith("list") or style == "list bullet":
            kind = "bullet"
        else:
            kind = "para"
        blocks.append({"kind": kind, "text": _clean_item(text) if kind == "bullet" else text})
        blocks.append({"kind": "blank", "text": ""})
    for table in getattr(document, "tables", []):
        for row in table.rows:
            cells = [(c.text or "").strip() for c in row.cells]
            cells = [c for c in cells if c]
            if cells:
                blocks.append({"kind": "para", "text": " | ".join(cells)})
                blocks.append({"kind": "blank", "text": ""})
    if blocks and _is_blank(blocks[-1]):
        blocks.pop()
    if not blocks:
        raise DocError("That Word file has no readable text.")
    return Document(path, "docx", [blocks])


def _load_text(path: str) -> Document:
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            content = fh.read()
    except Exception as e:
        raise DocError(f"Could not read the file: {e}") from e
    pages = [_split_blocks(chunk if chunk.strip() else " ")
             for chunk in content.split("\f")] if "\f" in content else [_split_blocks(content)]
    if not any(b["text"].strip() for page in pages for b in page):
        raise DocError("That file is empty.")
    return Document(path, "text", pages)


# --------------------------------------------------------------- humanize
class Cancelled(Exception):
    pass


def unit_count(doc: Document) -> int:
    """How many paragraphs a run will rewrite - what the UI counts on."""
    return sum(1 for page in doc.pages for b in page
               if not _is_blank(b) and b["text"].strip())


def humanize_document(engine, doc: Document, intensity: str,
                      on_progress: Optional[Callable] = None,
                      should_cancel: Optional[Callable] = None) -> List[List[dict]]:
    """Humanize paragraph by paragraph, page by page.

    on_progress(done, total, text) runs between units - it is called from a
    worker thread, so the GUI marshals it onto the main thread itself.
    """
    pages = doc.pages
    units: List[Tuple[int, int, str]] = []
    for pi, page in enumerate(pages):
        for bi in range(len(page)):
            if not _is_blank(page[bi]) and page[bi]["text"].strip():
                units.append((pi, bi, page[bi]["text"]))
    total = len(units) or 1
    done = 0

    for page_index, block_index, text in units:
        if should_cancel and should_cancel():
            raise Cancelled()
        block = doc.pages[page_index][block_index]
        done += 1
        if len(text.split()) < 4:
            # Too short to rewrite without mangling it - keep verbatim.
            if on_progress:
                on_progress(done, total, text)
            continue
        if block["kind"] == "head":
            out = text                              # headings stay untouched
        else:
            out = engine.humanize_text(text, intensity)
            if not out or not out.strip():
                out = text
        block["text"] = " ".join(out.split())
        if on_progress:
            on_progress(done, total, text)
    return pages


# -------------------------------------------------------------- measurement
_MEASURE_PX = 200          # measure once at this size, scale for real points
_FONT_FILES = {
    "regular": ("arial.ttf", "segoeui.ttf", "calibri.ttf", "DejaVuSans.ttf"),
    "bold": ("arialbd.ttf", "segoeuib.ttf", "calibrib.ttf", "DejaVuSans-Bold.ttf"),
}
_WIDTHS: dict = {}


def _approx_widths(bold: bool) -> dict:
    """Per-char widths in 1/1000 em, used only if no font file is readable."""
    table = {}
    for ch in "abcdefghijklmnopqrstuvwxyz":
        table[ch] = {"i": 222, "j": 222, "l": 222, "t": 278, "f": 278, "r": 333,
                     "m": 833, "w": 722}.get(ch, 556)
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        table[ch] = {"I": 278, "J": 500, "L": 556, "M": 833, "W": 944}.get(ch, 667)
    for ch in " .,:;'\"!|":
        table[ch] = {" ": 278, ".": 278, ",": 278, ":": 278, ";": 278,
                     "'": 191, "\"": 355, "!": 278, "|": 260}[ch]
    table.update({"-": 333, "_": 556, "(": 333, ")": 333, "[": 278, "]": 278,
                  "{": 334, "}": 334, "/": 278, "\\": 278, "?": 556, "=": 584,
                  "+": 584, "*": 389, "&": 667, "%": 889, "#": 556, "$": 556,
                  "@": 1015, "<": 584, ">": 584, "$": 556, "~": 584, "`": 333,
                  "@": 1015})
    table["1"], table["0"] = 556, 556
    for d in "23456789":
        table[d] = 556
    if bold:
        for ch, w in table.items():
            if ch != " ":
                table[ch] = min(1000, int(w * 1.06))
        table["i"] = table["j"] = table["l"] = 278
    return table


def _measure_font(kind: str):
    from PIL import ImageFont
    for name in _FONT_FILES[kind]:
        try:
            return ImageFont.truetype(name, _MEASURE_PX)
        except Exception:
            continue
    return None


def _measure(text: str, size: float, bold: bool) -> float:
    """Advance width in points. Arial metrics match the base-14 Helvetica
    widths, so the wrap we compute is what the PDF viewer will render."""
    key = "font" if bold else "approx"
    cached = _WIDTHS.get(key)
    if cached is None:
        cached = _WIDTHS[key] = _measure_font("bold" if bold else "regular")
    if cached is not None:
        try:
            return cached.getlength(text) * size / _MEASURE_PX
        except Exception:
            pass
    table = _WIDTHS.get(("t", bold))
    if table is None:
        table = _WIDTHS[("t", bold)] = _approx_widths(bold)
    total = sum(table.get(ch, 556) for ch in text)
    return total * size / 1000.0


# ----------------------------------------------------------------- wrapping
def _wrap(text: str, width: float, size: float, bold: bool) -> List[str]:
    lines: List[str] = []
    for hard in text.split("\n"):
        hard = hard.strip()
        if not hard:
            lines.append("")
            continue
        words = hard.split(" ")
        line = ""
        for word in words:
            if len(word.split()) == 1 and _measure(word, size, bold) > width:
                # single word longer than the column: cut it
                piece = ""
                for ch in word:
                    if piece and _measure(piece + ch, size, bold) > width:
                        lines.append(piece)
                        piece = ch
                    else:
                        piece += ch
                word = piece
                lines.append(word)
                word = ""
            candidate = f"{line} {word}".strip() if line else word
            if line and _measure(candidate, size, bold) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
    return lines or [""]


# --------------------------------------------------------------- PDF writer
BODY_SIZE = 10.5
BODY_LEAD = 15.0
HEAD_SIZE = 13.0
HEAD_LEAD = 18.0
MARGIN_X = 62.0
MARGIN_TOP = 66.0
MARGIN_BOTTOM = 58.0
BULLET = "\u2022"
BULLET_INDENT = 16.0
DEFAULT_SIZE = (612.0, 792.0)          # US Letter, PDF points


def _escape(raw: bytes) -> bytes:
    out = bytearray()
    for b in raw:
        if b in (0x28, 0x29, 0x5C):          # ( ) \
            out += b"\\" + bytes([b])
        elif b < 32 or b > 126:
            out += b"\\%03o" % b
        else:
            out.append(b)
    return bytes(out)


def _to_winansi(text: str) -> bytes:
    """Base-14 fonts use WinAnsi here, so encode to cp1252 (curly quotes and
    dashes included). Anything outside it - CJK, for instance - is dropped."""
    out = []
    for ch in text:
        try:
            out.append(ch.encode("cp1252"))
        except UnicodeEncodeError:
            if ch in ("\n", "\t"):
                out.append(b" ")
            else:
                out.append(b"?")
    return b"".join(out)


def _page_lines(blocks: Sequence[dict], width: float) -> List[dict]:
    """Blocks -> wrapped lines with font, size and vertical spacing."""
    lines: List[dict] = []
    for i, b in enumerate(blocks):
        if _is_blank(b):
            continue
        kind = b.get("kind")
        head = kind == "head"
        size = HEAD_SIZE if head else BODY_SIZE
        lead = HEAD_LEAD if head else BODY_LEAD
        prev = blocks[i - 1] if i else None
        was_blank = prev is None or _is_blank(prev)
        item = kind == "bullet"
        indent = BULLET_INDENT if item else 0.0
        wrapped = _wrap(b["text"], width - indent, size, head)
        gap = 0.0
        if head and was_blank:
            gap = 9.0
        elif head:
            gap = 7.0
        elif was_blank:
            gap = 5.0
        elif item:
            gap = 3.0            # breathe between consecutive list items
        last = len(wrapped) - 1
        for li, text in enumerate(wrapped):
            if not text:
                continue
            if item and li == 0:
                text = BULLET + "  " + text
            lines.append({"text": text, "bold": head, "size": size,
                          "lead": lead, "gap": gap if li == 0 else 0.0,
                          "last": li == last,
                          # only body text is justified; headings and list
                          # items stay flush left like a word processor
                          "justify": li == last and not head and not item,
                          "x_off": 0.0 if li == 0 else indent})
    return lines


def _content_stream(lines: Sequence[dict], width: float, height: float) -> bytes:
    """Text drawing ops for one page. Body lines are justified by nudging the
    word spacing (Tw), the way a word processor does, and each line ends with
    a drawn space so plain extractors keep the word break."""
    bottom = height - MARGIN_TOP
    top = MARGIN_BOTTOM
    parts: List[bytes] = [b"BT"]
    y = bottom
    for line in lines:
        y -= line["lead"] + line["gap"]
        if y < top:
            break
        font = b"/F2" if line["bold"] else b"/F1"
        parts.append(b"%s %s Tf" % (font, b"%.1f" % line["size"]))
        offset = line.get("x_off", 0.0)
        col = width - offset
        spaced = False
        if line.get("justify"):
            spaces = line["text"].count(" ")
            natural = _measure(line["text"], line["size"], line["bold"])
            if spaces and natural < col:
                extra = min(14.0, (col - natural) / spaces)
                if extra > 0.4:
                    parts.append(b"%.2f Tw" % extra)
                    spaced = True
        parts.append(b"1 0 0 1 %.2f %.2f Tm" % (MARGIN_X + offset, y))
        parts.append(b"(" + _escape(_to_winansi(line["text"])) + b") Tj")
        if spaced:
            parts.append(b"0 Tw")
        parts.append(b"( ) Tj")
    parts.append(b"ET")
    return b"\n".join(parts) + b"\n"


def _build_pdf(page_contents: Sequence[bytes],
               sizes: Sequence[Tuple[float, float]],
               title: str) -> bytes:
    objects: List[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)                     # object numbers are 1-based

    catalog = add(b"")                          # 1 - filled in at the end
    pages_obj = add(b"")                        # 2
    font_reg = add(b"<< /Font << /F1 %d 0 R /F2 %d 0 R >> >>"
                   % (add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                           b"/Encoding /WinAnsiEncoding >>"),
                      add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
                          b"/Encoding /WinAnsiEncoding >>")))
    info_obj = add(b"<< /Producer (Humanizer) /Title (%s) >>"
                   % _escape(_to_winansi(title)))

    page_ids = []
    for i, content in enumerate(page_contents):
        width, height = sizes[i] if i < len(sizes) else sizes[-1]
        width = float(width) or DEFAULT_SIZE[0]
        height = float(height) or DEFAULT_SIZE[1]
        stream = add(b"<< /Length %d /Filter /FlateDecode >>\nstream\n%s\nendstream"
                     % (len(content), zlib.compress(content, 6)))
        page = add(("<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %.2f %.2f] "
                    "/Resources %d 0 R /Contents %d 0 R >>"
                    % (pages_obj, width, height, font_reg, stream)).encode("latin-1"))
        page_ids.append(page)

    kids = b" ".join(b"%d 0 R" % p for p in page_ids)
    objects[catalog - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_obj
    objects[pages_obj - 1] = (b"<< /Type /Pages /Kids [%s] /Count %d >>"
                              % (kids, len(page_ids)))

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % num + body + b"\nendobj\n"
    xref = len(out)
    count = len(objects) + 1
    out += b"xref\n0 %d\n" % count
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (count, catalog, info_obj, xref))
    return bytes(out)


def layout_pdf(pages: Sequence[Sequence[dict]],
               sizes: Sequence[Optional[Tuple[float, float]]]
               ) -> Tuple[List[bytes], List[Tuple[float, float]]]:
    """Flow blocks onto pages, keeping one source page per output page."""
    contents: List[bytes] = []
    used: List[Tuple[float, float]] = []
    if not pages:
        pages = [[]]
    for i, blocks in enumerate(pages):
        width, height = (sizes[i] if i < len(sizes) and sizes[i]
                         else DEFAULT_SIZE)
        width, height = float(width), float(height)
        col = width - 2 * MARGIN_X
        lines = _page_lines(blocks, col)
        used.append((width, height))
        if not lines:
            contents.append(_content_stream([], col, height))
            continue
        # split on page overflow; a source page may need more than one page
        chunk: List[dict] = []
        used_height = 0.0
        for line in lines:
            step = line["lead"] + line["gap"]
            if used_height + step > height - MARGIN_TOP - MARGIN_BOTTOM and chunk:
                contents.append(_content_stream(chunk, col, height))
                used.append((width, height))
                chunk, used_height = [], 0.0
                line = dict(line, gap=0.0)
                step = line["lead"]
            chunk.append(line)
            used_height += step
        contents.append(_content_stream(chunk, col, height))
    return contents, used


def write_pdf(path: str, pages: Sequence[Sequence[dict]],
              sizes: Sequence[Optional[Tuple[float, float]]] = None,
              title: str = "Humanized document") -> str:
    """Write humanized pages to a new PDF. Returns the path written."""
    sizes = list(sizes or [])
    contents, used = layout_pdf(pages, sizes)
    data = _build_pdf(contents, used, title)
    folder = os.path.dirname(os.path.abspath(path))
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


# -------------------------------------------------------------------- misc
def suggested_name(doc_path: str, suffix: str = "_humanized") -> str:
    stem = os.path.splitext(os.path.basename(doc_path))[0] or "document"
    return f"{stem}{suffix}.pdf"


def save_report(doc: Document, pages: Sequence[Sequence[dict]], out_path: str) -> str:
    """Write the humanized document, matching the source page size for PDFs."""
    if doc.kind == "pdf":
        return write_pdf(out_path, pages, doc.sizes, title=doc.name)
    return write_pdf(out_path, pages, None, title=doc.name)