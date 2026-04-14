import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from extractors import extract_text, chunk_text, SUPPORTED_EXTS
import logging

# -------- SUPPRESS NOISY LOGGERS --------
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# -------- CONFIG --------
DOC_FOLDER     = r"D:\coding\college"
DATA_DIR       = "data"
CHECK_INTERVAL = 60
MAX_FILE_SIZE  = 50 * 1024 * 1024

EMBED_PATH = os.path.join(DATA_DIR, "embeddings.npy")
META_PATH  = os.path.join(DATA_DIR, "documents.pkl")

SKIP_PREFIXES = ("~$", ".", "__")
SKIP_SUFFIXES = (".tmp", ".lock", ".lnk", ".ds_store")

os.makedirs(DATA_DIR, exist_ok=True)

# -------- LOAD MODEL --------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
model = SentenceTransformer(MODEL_PATH, local_files_only=True)
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
    name_lower = filename.lower()
    if any(name_lower.startswith(p) for p in SKIP_PREFIXES):
        return True
    if any(name_lower.endswith(s) for s in SKIP_SUFFIXES):
        return True
    ext = os.path.splitext(name_lower)[1]
    if ext not in SUPPORTED_EXTS:
        return True
    return False


def safe_encode(text: str, path: str):
    try:
        return model.encode(text)
    except Exception as e:
        print(f"  [ENCODE] Failed to encode '{path}': {e}")
        return None


# -------- INCREMENTAL INDEX --------

def incremental_index():
    global embeddings, metadata

    print("\nScanning folder for changes...")

    # With chunking, multiple metadata rows share the same path.
    # dict comprehension naturally deduplicates — last mtime wins.
    # This is fine since all chunks of a file share the same mtime.
    indexed_files = {item["path"]: item["mtime"] for item in metadata}

    # Use a SET — O(1) lookup when checking deleted files
    current_files = set()
    new_vectors   = []

    for root, _, files in os.walk(DOC_FOLDER):
        for file in files:

            if should_skip(file):
                continue

            path = os.path.join(root, file)
            current_files.add(path)

            try:
                mtime = os.path.getmtime(path)
            except OSError as e:
                print(f"  [SKIP] Cannot stat '{path}': {e}")
                continue

            # -------- NEW FILE --------
            if path not in indexed_files:
                text     = extract_text(path)
                filename = os.path.splitext(os.path.basename(path))[0]
                combined = f"{filename}\n{text}".strip()

                if combined.strip():
                    chunks = chunk_text(combined)
                    for chunk_idx, chunk in enumerate(chunks):
                        vec = safe_encode(chunk, path)
                        if vec is not None:
                            new_vectors.append(vec)
                            metadata.append({
                                "path":  path,
                                "mtime": mtime,
                                "chunk": chunk_idx
                            })
                    print(f"  [NEW] Indexed: {path} ({len(chunks)} chunks)")

            # -------- MODIFIED FILE --------
            elif indexed_files[path] != mtime:
                text     = extract_text(path)
                filename = os.path.splitext(os.path.basename(path))[0]
                combined = f"{filename}\n{text}".strip()

                if combined.strip():
                    # remove ALL existing chunks for this file first
                    existing_indices = [
                        i for i, m in enumerate(metadata)
                        if m["path"] == path
                    ]
                    if existing_indices:
                        embeddings = np.delete(embeddings, existing_indices, axis=0)
                        metadata   = [m for i, m in enumerate(metadata)
                                      if i not in existing_indices]

                    chunks = chunk_text(combined)
                    for chunk_idx, chunk in enumerate(chunks):
                        vec = safe_encode(chunk, path)
                        if vec is not None:
                            new_vectors.append(vec)
                            metadata.append({
                                "path":  path,
                                "mtime": mtime,
                                "chunk": chunk_idx
                            })
                    print(f"  [UPDATED] Re-indexed: {path} ({len(chunks)} chunks)")

    # -------- DELETE HANDLING --------
    deleted_indices = [
        i for i, item in enumerate(metadata)
        if item["path"] not in current_files
    ]

    if deleted_indices:
        # Save paths BEFORE rebuilding metadata so the print is accurate
        deleted_paths = [metadata[i]["path"] for i in deleted_indices]
        embeddings    = np.delete(embeddings, deleted_indices, axis=0)
        metadata      = [m for i, m in enumerate(metadata)
                         if i not in deleted_indices]
        # deduplicate — multiple chunks produce the same path, print it once
        for path in sorted(set(deleted_paths)):
            print(f"  [REMOVED] {path}")

    # -------- ADD NEW EMBEDDINGS --------
    if new_vectors:
        new_array  = np.array(new_vectors)
        embeddings = new_array if embeddings.size == 0 \
                     else np.vstack((embeddings, new_array))

    # -------- SAVE INDEX --------
    np.save(EMBED_PATH, embeddings)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index updated. Total chunks indexed: {len(metadata)}")