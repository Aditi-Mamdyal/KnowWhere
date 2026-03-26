import os
import pickle
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document
import logging

# -------- SUPPRESS NOISY LOGGERS --------
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

# -------- CONFIG --------
DOC_FOLDER     = r"D:\coding\college"
DATA_DIR       = "data"
CHECK_INTERVAL = 60         # fixed typo: was CHECK_INTERAVL
MAX_FILE_SIZE  = 50 * 1024 * 1024   # 50 MB — skip files larger than this

EMBED_PATH = os.path.join(DATA_DIR, "embeddings.npy")
META_PATH  = os.path.join(DATA_DIR, "documents.pkl")

# Temp/lock file prefixes and suffixes to skip unconditionally
SKIP_PREFIXES = ("~$", ".", "__")
SKIP_SUFFIXES = (".tmp", ".lock", ".lnk", ".ds_store")

# Supported extensions
SUPPORTED_EXTS = {".txt", ".pdf", ".docx"}

os.makedirs(DATA_DIR, exist_ok=True)

# -------- LOAD MODEL --------
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

# -------- LOAD EXISTING INDEX --------
if os.path.exists(EMBED_PATH):
    embeddings = np.load(EMBED_PATH)
else:
    embeddings = np.empty((0, 384))

if os.path.exists(META_PATH):
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
else:
    metadata = []


# -------- HELPERS --------

def should_skip(filename: str) -> bool:
    """Return True if this filename should never be processed."""
    name_lower = filename.lower()
    if any(name_lower.startswith(p) for p in SKIP_PREFIXES):
        return True
    if any(name_lower.endswith(s) for s in SKIP_SUFFIXES):
        return True
    ext = os.path.splitext(name_lower)[1]
    if ext not in SUPPORTED_EXTS:
        return True
    return False


def extract_text_txt(filepath: str) -> str:
    """Extract text from a plain-text file, trying multiple encodings."""
    for encoding in ("utf-8", "utf-16", "latin-1", "cp1252"):
        try:
            with open(filepath, "r", encoding=encoding, errors="strict") as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    # Last-resort: ignore undecodable bytes
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text_pdf(filepath: str) -> str:
    """Extract text from a PDF, skipping unreadable pages gracefully."""
    text_parts = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as page_err:
                    print(f"  [PDF] Skipping page {page_num} in '{filepath}': {page_err}")
    except Exception as e:
        print(f"  [PDF] Cannot open '{filepath}': {e}")
    return "\n".join(text_parts)


def extract_text_docx(filepath: str) -> str:
    """Extract text from a .docx file."""
    try:
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"  [DOCX] Cannot read '{filepath}': {e}")
        return ""


def extract_text(filepath: str) -> str:
    """
    Dispatch to the right extractor based on extension.
    Returns an empty string on any unrecoverable error so the
    caller can safely skip the file.
    """
    # Guard: file must still exist (race condition between scan and read)
    if not os.path.isfile(filepath):
        print(f"  [SKIP] File disappeared before reading: '{filepath}'")
        return ""

    # Guard: skip oversized files to avoid OOM
    try:
        size = os.path.getsize(filepath)
    except OSError:
        return ""
    if size == 0:
        print(f"  [SKIP] Empty file: '{filepath}'")
        return ""
    if size > MAX_FILE_SIZE:
        print(f"  [SKIP] File too large ({size // (1024*1024)} MB): '{filepath}'")
        return ""

    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".txt":
            return extract_text_txt(filepath)
        if ext == ".pdf":
            return extract_text_pdf(filepath)
        if ext == ".docx":
            return extract_text_docx(filepath)
    except Exception as e:
        # Catch-all so a single bad file never kills the indexer
        print(f"  [ERROR] Unexpected error reading '{filepath}': {e}")

    return ""


def safe_encode(text: str, path: str):
    """Encode text to a vector; return None on failure."""
    try:
        return model.encode(text)
    except Exception as e:
        print(f"  [ENCODE] Failed to encode '{path}': {e}")
        return None


# -------- INCREMENTAL INDEX --------

def incremental_index():
    global embeddings, metadata

    print("\nScanning folder for changes...")

    indexed_files = {item["path"]: item["mtime"] for item in metadata}
    current_files = []
    new_vectors   = []

    for root, _, files in os.walk(DOC_FOLDER):
        for file in files:

            if should_skip(file):
                continue

            path = os.path.join(root, file)
            current_files.append(path)

            try:
                mtime = os.path.getmtime(path)
            except OSError as e:
                print(f"  [SKIP] Cannot stat '{path}': {e}")
                continue

            # -------- NEW FILE --------
            if path not in indexed_files:
                text = extract_text(path)
                if text.strip():
                    vec = safe_encode(text, path)
                    if vec is not None:
                        new_vectors.append(vec)
                        metadata.append({"path": path, "mtime": mtime})
                        print(f"  [NEW] Indexed: {path}")

            # -------- MODIFIED FILE --------
            elif indexed_files[path] != mtime:
                text = extract_text(path)
                if text.strip():
                    vec = safe_encode(text, path)
                    if vec is not None:
                        idx = next(
                            (i for i, m in enumerate(metadata) if m["path"] == path),
                            None
                        )
                        if idx is not None:
                            embeddings[idx]        = vec
                            metadata[idx]["mtime"] = mtime
                            print(f"  [UPDATED] Re-indexed: {path}")

    # -------- DELETE HANDLING --------
    deleted_indices = [
        i for i, item in enumerate(metadata)
        if item["path"] not in current_files
    ]

    if deleted_indices:
        embeddings = np.delete(embeddings, deleted_indices, axis=0)
        metadata   = [m for i, m in enumerate(metadata) if i not in deleted_indices]
        for i in deleted_indices:
            print(f"  [REMOVED] {metadata[i]['path'] if i < len(metadata) else i}")

    # -------- ADD NEW EMBEDDINGS --------
    if new_vectors:
        new_array = np.array(new_vectors)
        embeddings = new_array if embeddings.size == 0 else np.vstack((embeddings, new_array))

    # -------- SAVE INDEX --------
    np.save(EMBED_PATH, embeddings)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    total = len(metadata)
    print(f"Index updated. Total documents indexed: {total}")


# -------- MAIN LOOP --------
while True:
    try:
        incremental_index()
    except Exception as e:
        # The loop itself must never crash — log and retry next cycle
        print(f"\n[CRITICAL] incremental_index() raised an unexpected error: {e}")
        print("Continuing to next scan cycle...\n")

    print(f"\nNext scan in {CHECK_INTERVAL} seconds...\n")
    time.sleep(CHECK_INTERVAL)