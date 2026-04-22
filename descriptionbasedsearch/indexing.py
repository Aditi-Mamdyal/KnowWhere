import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from extractors import extract_text, SUPPORTED_EXTS   # NO chunk_text import
import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ── absolute root path — works from VS Code, .bat, desktop shortcut ──────────
_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHECK_INTERVAL = 60
MAX_FILE_SIZE  = 50 * 1024 * 1024        # 50 MB — skips large textbooks

DATA_DIR    = os.path.join(_ROOT, "data")
EMBED_PATH  = os.path.join(DATA_DIR, "embeddings.npy")
META_PATH   = os.path.join(DATA_DIR, "documents.pkl")
MODEL_PATH  = os.path.join(_ROOT, "models", "all-MiniLM-L6-v2")
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")

SKIP_PREFIXES = ("~$", ".", "__")
SKIP_SUFFIXES = (".tmp", ".lock", ".lnk", ".ds_store")

os.makedirs(DATA_DIR, exist_ok=True)


# ── CONFIG HELPERS ─────────────────────────────────────────────────────────────

def get_doc_folder() -> str:
    """Returns saved DOC_FOLDER from config.json, or None if not set yet."""
    if not os.path.exists(CONFIG_FILE):
        return None
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("doc_folder")
    except Exception:
        return None


def save_doc_folder(path: str):
    """Save folder choice to config.json and update global DOC_FOLDER."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"doc_folder": path}, f, indent=2)
    global DOC_FOLDER
    DOC_FOLDER = path
    print(f"[CONFIG] DOC_FOLDER set to: {path}")


def is_setup_complete() -> bool:
    """True if folder has been configured."""
    return get_doc_folder() is not None


# ── SET DOC_FOLDER ────────────────────────────────────────────────────────────
DOC_FOLDER = get_doc_folder() or r"D:\coding\college"

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
model = SentenceTransformer(MODEL_PATH, local_files_only=True)
print("Model loaded.")

# ── LOAD EXISTING INDEX ───────────────────────────────────────────────────────
if os.path.exists(EMBED_PATH):
    embeddings = np.load(EMBED_PATH)
else:
    embeddings = np.empty((0, 384))

if os.path.exists(META_PATH):
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
else:
    metadata = []


# ── HELPERS ───────────────────────────────────────────────────────────────────

def should_skip(filename: str) -> bool:
    name_lower = filename.lower()
    if any(name_lower.startswith(p) for p in SKIP_PREFIXES):
        return True
    if any(name_lower.endswith(s) for s in SKIP_SUFFIXES):
        return True
    if os.path.splitext(name_lower)[1] not in SUPPORTED_EXTS:
        return True
    return False


def safe_encode(text: str, path: str):
    try:
        return model.encode(text)
    except Exception as e:
        print(f"  [ENCODE] Failed to encode '{path}': {e}")
        return None


# ── INCREMENTAL INDEX — one vector per document ───────────────────────────────

def incremental_index():
    """
    Scans DOC_FOLDER and updates the index.
    One vector per document — no chunking.

    Uses phase-separated logic to avoid index corruption:
      Phase 1 — scan folder, collect what changed (no writes)
      Phase 2 — delete stale entries all at once
      Phase 3 — append new vectors
      Phase 4 — save to disk
    """
    global embeddings, metadata

    if not DOC_FOLDER or not os.path.isdir(DOC_FOLDER):
        print(f"[INDEX] DOC_FOLDER not set or missing: {DOC_FOLDER}")
        return

    print("\nScanning folder for changes...")

    # one entry per file: path -> mtime
    indexed_files = {item["path"]: item["mtime"] for item in metadata}
    current_files = set()

    paths_to_add    = {}   # path -> (mtime, vector)
    paths_to_update = {}   # path -> (mtime, vector)

    # ── PHASE 1: SCAN — collect only, no writes ───────────────────────────────
    for root, _, files in os.walk(DOC_FOLDER):
        for file in files:
            if should_skip(file):
                continue

            path = os.path.join(root, file)
            current_files.add(path)

            try:
                mtime = os.path.getmtime(path)
                size  = os.path.getsize(path)
            except OSError:
                continue

            if size == 0 or size > MAX_FILE_SIZE:
                continue

            if path not in indexed_files:
                # new file
                text     = extract_text(path)
                filename = os.path.splitext(os.path.basename(path))[0]
                combined = f"{filename}\n{text}".strip()
                if combined:
                    vec = safe_encode(combined, path)
                    if vec is not None:
                        paths_to_add[path] = (mtime, vec)
                        print(f"  [NEW] Indexed: {path}")

            elif indexed_files[path] != mtime:
                # modified file
                text     = extract_text(path)
                filename = os.path.splitext(os.path.basename(path))[0]
                combined = f"{filename}\n{text}".strip()
                if combined:
                    vec = safe_encode(combined, path)
                    if vec is not None:
                        paths_to_update[path] = (mtime, vec)
                        print(f"  [UPDATED] Re-indexed: {path}")

    # ── PHASE 2: DELETE ALL AT ONCE ───────────────────────────────────────────
    paths_to_remove = (
        {p for p in indexed_files if p not in current_files}  # deleted from disk
        | set(paths_to_update.keys())                          # being replaced
    )

    if paths_to_remove:
        indices = [i for i, m in enumerate(metadata)
                   if m["path"] in paths_to_remove]
        if indices:
            embeddings = np.delete(embeddings, indices, axis=0)
            metadata   = [m for i, m in enumerate(metadata)
                          if i not in set(indices)]
        deleted_from_disk = {p for p in paths_to_remove if p not in paths_to_update}
        for p in sorted(deleted_from_disk):
            print(f"  [REMOVED] {p}")

    # ── PHASE 3: APPEND ───────────────────────────────────────────────────────
    new_vecs = []
    new_meta = []
    for path, (mtime, vec) in {**paths_to_add, **paths_to_update}.items():
        new_vecs.append(vec)
        new_meta.append({"path": path, "mtime": mtime})

    if new_vecs:
        arr        = np.array(new_vecs)
        embeddings = arr if embeddings.size == 0 \
                     else np.vstack((embeddings, arr))
        metadata.extend(new_meta)

    # ── PHASE 4: SAVE ─────────────────────────────────────────────────────────
    np.save(EMBED_PATH, embeddings)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index updated. Total documents indexed: {len(metadata)}")