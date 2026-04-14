#understanding phtos in docs needs clip pipeline which is still pending

import os
import logging

# -------- SUPPRESS NOISY LOGGERS --------
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("pytesseract").setLevel(logging.ERROR)

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------- ALL SUPPORTED EXTENSIONS --------
SUPPORTED_EXTS = {
    # Text-based
    ".txt", ".md", ".rtf",
    # Spreadsheets
    ".xlsx", ".xls", ".csv", ".ods",
    # Presentations
    ".pptx", ".ppt", ".odp",
    # Word / rich text
    ".docx", ".odt",
    # PDFs (normal + scanned)
    ".pdf",
    # Email
    ".eml", ".msg",
    # Web / code / project files
    ".html", ".htm", ".xml", ".json",
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".cs",
    ".md", ".rst", ".yaml", ".yml", ".toml", ".ini", ".cfg",
}

MAX_FILE_SIZE = 100 * 1024 * 1024


# =============================================================================
# 1. TEXT-BASED DOCUMENTS
# =============================================================================
# .txt, .md, .rtf, .rst, .yaml, .yml, .toml, .ini, .cfg
# Plain text files — just read them. Try multiple encodings since corporate machines may have files saved in various Windows encodings.

def extract_text_txt(filepath: str) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1", "cp1252"):
        try:
            with open(filepath, "r", encoding=encoding, errors="strict") as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# =============================================================================
# 2. SPREADSHEET AND DATA FILES
# =============================================================================
# .xlsx, .xls  — openpyxl / xlrd
# .csv         — plain text read
# .ods         — odfpy
#
# Strategy: read every cell from every sheet and join into one text blob.
# Column headers are included so queries like "find spreadsheet with revenue column" still work.

def extract_text_xlsx(filepath: str) -> str:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        parts = []
        for sheet in wb.worksheets:
            parts.append(f"Sheet: {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                row_text = " | ".join(str(cell) for cell in row if cell is not None)
                if row_text.strip():
                    parts.append(row_text)
        return "\n".join(parts)
    except Exception as e:
        print(f"  [XLSX] Cannot read '{filepath}': {e}")
        return ""


def extract_text_xls(filepath: str) -> str:
    # .xls is the old Excel format (pre-2007) — needs xlrd
    try:
        import xlrd
        wb = xlrd.open_workbook(filepath)
        parts = []
        for sheet in wb.sheets():
            parts.append(f"Sheet: {sheet.name}")
            for row_idx in range(sheet.nrows):
                row = sheet.row_values(row_idx)
                row_text = " | ".join(str(v) for v in row if str(v).strip())
                if row_text.strip():
                    parts.append(row_text)
        return "\n".join(parts)
    except Exception as e:
        print(f"  [XLS] Cannot read '{filepath}': {e}")
        return ""


def extract_text_csv(filepath: str) -> str:
    # CSV is plain text — just read it directly like a txt file
    return extract_text_txt(filepath)


def extract_text_ods(filepath: str) -> str:
    # .ods is LibreOffice Calc format — uses odfpy
    try:
        from odf.opendocument import load
        from odf.table import Table, TableRow, TableCell
        from odf.text import P

        doc = load(filepath)
        parts = []
        for sheet in doc.spreadsheet.getElementsByType(Table):
            parts.append(f"Sheet: {sheet.getAttribute('name')}")
            for row in sheet.getElementsByType(TableRow):
                cells = []
                for cell in row.getElementsByType(TableCell):
                    ps = cell.getElementsByType(P)
                    cell_text = " ".join(
                        "".join(str(n) for n in p.childNodes) for p in ps
                    )
                    if cell_text.strip():
                        cells.append(cell_text.strip())
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    except Exception as e:
        print(f"  [ODS] Cannot read '{filepath}': {e}")
        return ""


# =============================================================================
# 3. PRESENTATION FILES
# =============================================================================
# .pptx — python-pptx
# .odp  — odfpy
#
# Strategy: extract text from every slide, including:
#   - Slide title
#   - All text boxes / bullet points
#   - Speaker notes (often contain more detail than the slide itself)
#   - Chart titles and axis labels (if present)
# This means a search for "Q3 revenue projections" will find the presentation even if that phrase is only in the speaker notes.

def extract_text_pptx(filepath: str) -> str:
    try:
        from pptx import Presentation
        from pptx.util import Pt

        prs = Presentation(filepath)
        parts = []

        for slide_num, slide in enumerate(prs.slides, start=1):
            parts.append(f"\n--- Slide {slide_num} ---")

            # all text shapes (titles, text boxes, bullet points)
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            parts.append(text)

                # chart titles (charts have no text_frame but have a title)
                if shape.has_chart:
                    chart = shape.chart
                    if chart.has_title:
                        parts.append(f"Chart: {chart.chart_title.text_frame.text}")

            # speaker notes — often contain rich context
            if slide.has_notes_slide:
                notes_text = slide.notes_slide.notes_text_frame.text.strip()
                if notes_text:
                    parts.append(f"Notes: {notes_text}")

        return "\n".join(parts)
    except Exception as e:
        print(f"  [PPTX] Cannot read '{filepath}': {e}")
        return ""


def extract_text_odp(filepath: str) -> str:
    # .odp is LibreOffice Impress format
    try:
        from odf.opendocument import load
        from odf.text import P

        doc = load(filepath)
        parts = []
        for p in doc.getElementsByType(P):
            text = "".join(str(n) for n in p.childNodes).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    except Exception as e:
        print(f"  [ODP] Cannot read '{filepath}': {e}")
        return ""


# =============================================================================
# 4. PDF DOCUMENTS
# =============================================================================
# .pdf — pdfplumber for text-based PDFs
#        pytesseract OCR fallback for scanned PDFs
#
# Strategy:
#   1. Try pdfplumber first (fast, accurate for digital PDFs)
#   2. If a page returns no text (scanned page), fall back to OCR on that page
#   3. This handles mixed PDFs (some digital pages, some scanned pages)

def extract_text_pdf(filepath: str) -> str:
    try:
        import pdfplumber
        from PIL import Image
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

        text_parts = []

        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text()

                    if page_text and page_text.strip():
                        # normal digital page — use pdfplumber text directly
                        text_parts.append(page_text)
                    else:
                        # no text found — page is likely scanned, try OCR
                        print(f"  [PDF] Page {page_num} has no text — trying OCR...")
                        try:
                            img = page.to_image(resolution=300).original
                            ocr_text = pytesseract.image_to_string(img)
                            if ocr_text.strip():
                                text_parts.append(ocr_text)
                        except Exception as ocr_err:
                            print(f"  [PDF] OCR failed on page {page_num}: {ocr_err}")

                except Exception as page_err:
                    print(f"  [PDF] Skipping page {page_num} in '{filepath}': {page_err}")

        return "\n".join(text_parts)

    except Exception as e:
        print(f"  [PDF] Cannot open '{filepath}': {e}")
        return ""


# =============================================================================
# 5. EMAIL DOCUMENTS
# =============================================================================
# .eml — standard email format (Outlook, Thunderbird, Gmail export)
# .msg — Microsoft Outlook proprietary format
#
# Strategy: extract subject, sender, recipients, body text.
# Attachments are NOT extracted here (that would require recursive indexing).
# The email metadata (who sent it, subject) is included in the text so you can search "email from John about budget approval".

def extract_text_eml(filepath: str) -> str:
    try:
        import email
        from email import policy

        with open(filepath, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        parts = []

        # metadata — useful for searching by sender/subject
        parts.append(f"Subject: {msg.get('subject', '')}")
        parts.append(f"From: {msg.get('from', '')}")
        parts.append(f"To: {msg.get('to', '')}")
        parts.append(f"Date: {msg.get('date', '')}")

        # body — walk through all parts of the email
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                body = part.get_content()
                if body and body.strip():
                    parts.append(body)
            elif content_type == "text/html":
                # strip HTML tags for plain text
                html = part.get_content()
                if html:
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html, "html.parser")
                        parts.append(soup.get_text(separator=" "))
                    except Exception:
                        parts.append(html)  # fallback: raw HTML

        return "\n".join(parts)
    except Exception as e:
        print(f"  [EML] Cannot read '{filepath}': {e}")
        return ""


def extract_text_msg(filepath: str) -> str:
    # .msg is Microsoft Outlook's proprietary format — needs extract-msg
    # install: pip install extract-msg
    try:
        import extract_msg
        msg = extract_msg.Message(filepath)
        parts = [
            f"Subject: {msg.subject or ''}",
            f"From: {msg.sender or ''}",
            f"To: {msg.to or ''}",
            f"Date: {msg.date or ''}",
            msg.body or "",
        ]
        return "\n".join(p for p in parts if p.strip())
    except ImportError:
        print("  [MSG] extract-msg not installed. Run: pip install extract-msg")
        return ""
    except Exception as e:
        print(f"  [MSG] Cannot read '{filepath}': {e}")
        return ""


# =============================================================================
# 6. WORD / RICH TEXT DOCUMENTS
# =============================================================================
# .docx — python-docx (same as your original)
# .odt  — odfpy (LibreOffice Writer)
# .rtf  — strip RTF control codes, read as plain text

def extract_text_docx(filepath: str) -> str:
    try:
        from docx import Document
        doc = Document(filepath)
        parts = []

        # paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)

        # tables — important for corporate docs that store data in tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    parts.append(row_text)

        return "\n".join(parts)
    except Exception as e:
        print(f"  [DOCX] Cannot read '{filepath}': {e}")
        return ""


def extract_text_odt(filepath: str) -> str:
    try:
        from odf.opendocument import load
        from odf.text import P
        doc = load(filepath)
        parts = []
        for p in doc.getElementsByType(P):
            text = "".join(str(n) for n in p.childNodes).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    except Exception as e:
        print(f"  [ODT] Cannot read '{filepath}': {e}")
        return ""


def extract_text_rtf(filepath: str) -> str:
    # RTF files have lots of control codes like \rtf1\ansi\deff0
    # We strip them with a simple regex to get the readable text
    try:
        import re
        with open(filepath, "rb") as f:
            raw = f.read().decode("latin-1", errors="ignore")
        # remove RTF control words and groups
        text = re.sub(r'\\\w+', ' ', raw)
        text = re.sub(r'[{}]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        print(f"  [RTF] Cannot read '{filepath}': {e}")
        return ""

# =============================================================================
# 8. WEB AND PROJECT FILES
# =============================================================================
# .html / .htm — strip tags, extract readable text
# .xml         — strip tags
# .json        — flatten all string values
# .py .js .ts .java .cpp .c .cs — source code as-is
# .rst .md     — plain text (already handled by extract_text_txt)
# .yaml .yml .toml .ini .cfg — plain text (already handled by extract_text_txt)

def extract_text_html(filepath: str) -> str:
    try:
        from bs4 import BeautifulSoup
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
        # remove script and style blocks — they add noise
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"  [HTML] Cannot read '{filepath}': {e}")
        return ""


def extract_text_xml(filepath: str) -> str:
    try:
        from bs4 import BeautifulSoup
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "xml")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"  [XML] Cannot read '{filepath}': {e}")
        return ""


def extract_text_json(filepath: str) -> str:
    # Flatten all string values from JSON so they're searchable
    try:
        import json

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        def flatten(obj, parts):
            if isinstance(obj, str):
                parts.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    flatten(v, parts)
            elif isinstance(obj, list):
                for item in obj:
                    flatten(item, parts)

        parts = []
        flatten(data, parts)
        return "\n".join(parts)
    except Exception as e:
        print(f"  [JSON] Cannot read '{filepath}': {e}")
        return ""


def extract_text_code(filepath: str) -> str:
    # Source code files — read as plain text
    # Comments and strings are the most semantically useful parts
    return extract_text_txt(filepath)


# =============================================================================
# MAIN DISPATCHER — extract_text()
# =============================================================================
# This is the single function imported by indexing.py.
# It checks the file's extension and calls the right extractor.
# Returns empty string on any failure so the caller safely skips the file.

def extract_text(filepath: str) -> str:

    # guard: file must still exist
    if not os.path.isfile(filepath):
        print(f"  [SKIP] File disappeared: '{filepath}'")
        return ""

    # guard: size check
    try:
        size = os.path.getsize(filepath)
    except OSError:
        return ""
    if size == 0:
        print(f"  [SKIP] Empty file: '{filepath}'")
        return ""
    if size > MAX_FILE_SIZE:
        print(f"  [SKIP] Too large ({size // (1024*1024)} MB): '{filepath}'")
        return ""

    ext = os.path.splitext(filepath)[1].lower()

    try:
        # ---- text / markup / config ----
        if ext in {".txt", ".md", ".rst", ".yaml", ".yml",
                   ".toml", ".ini", ".cfg"}:
            return extract_text_txt(filepath)

        # ---- RTF ----
        if ext == ".rtf":
            return extract_text_rtf(filepath)

        # ---- spreadsheets ----
        if ext == ".xlsx":
            return extract_text_xlsx(filepath)
        if ext == ".xls":
            return extract_text_xls(filepath)
        if ext == ".csv":
            return extract_text_csv(filepath)
        if ext == ".ods":
            return extract_text_ods(filepath)

        # ---- presentations ----
        if ext == ".pptx":
            return extract_text_pptx(filepath)
        if ext in {".odp", ".ppt"}:
            return extract_text_odp(filepath)

        # ---- word / rich text ----
        if ext == ".docx":
            return extract_text_docx(filepath)
        if ext == ".odt":
            return extract_text_odt(filepath)

        # ---- PDF (digital + scanned) ----
        if ext == ".pdf":
            return extract_text_pdf(filepath)

        # ---- email ----
        if ext == ".eml":
            return extract_text_eml(filepath)
        if ext == ".msg":
            return extract_text_msg(filepath)

        # ---- web / data ----
        if ext in {".html", ".htm"}:
            return extract_text_html(filepath)
        if ext == ".xml":
            return extract_text_xml(filepath)
        if ext == ".json":
            return extract_text_json(filepath)

        # ---- source code ----
        if ext in {".py", ".js", ".ts", ".java", ".cpp",
                   ".c", ".cs"}:
            return extract_text_code(filepath)

    except Exception as e:
        print(f"  [ERROR] Unexpected error reading '{filepath}': {e}")

    return ""

# =============================================================================
# TEXT CHUNKING
# =============================================================================

def chunk_text(text: str, chunk_size: int = 400,
               overlap: int = 50) -> list:
    words = text.split()

    # if text is short enough, no chunking needed — return as single chunk
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start  = 0

    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap   # move forward by (chunk_size - overlap) so next chunk starts 50 words before current chunk ended                            

    return chunks