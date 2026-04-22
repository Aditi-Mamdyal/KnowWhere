import os
import re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

_ROOT      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_ROOT, "models", "all-MiniLM-L6-v2")
EMBED_PATH = os.path.join(_ROOT, "data", "embeddings.npy")
META_PATH  = os.path.join(_ROOT, "data", "documents.pkl")

model = SentenceTransformer(MODEL_PATH, local_files_only=True)


def _load_doc_index():
    if not os.path.exists(EMBED_PATH) or not os.path.exists(META_PATH):
        return np.empty((0, 384)), []
    emb = np.load(EMBED_PATH)
    # FIX: use context manager so the file handle is always closed,
    # prevents Windows file-lock issues that blocked re-saves after indexing.
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return emb, meta

embeddings, metadata = _load_doc_index()

image_search_available = False
try:
    from image_search import search_images, reload_index as _reload_image_index
    image_search_available = True
except ImportError:
    _reload_image_index = None

try:
    from auth import log_search
    _auth_available = True
except ImportError:
    _auth_available = False


def cosine_similarity(a, b):
    n = norm(a) * norm(b)
    return float(np.dot(a, b) / n) if n != 0 else 0.0


def reload_doc_index():
    """
    Reload the document index from disk into this module's globals.
    Called by indexing_service after every successful index run so that
    searches immediately reflect newly added / modified / deleted files.
    """
    global embeddings, metadata
    embeddings, metadata = _load_doc_index()
    print(f"[SEARCH] Doc index reloaded — {len(metadata)} documents in memory.")

    # Also reload image index if available
    if _reload_image_index is not None:
        _reload_image_index()
        print("[SEARCH] Image index reloaded.")


def _tokenize(text):
    tokens = re.split(r'[\s_\-\.\,\/\\]+', text.lower())
    result = set()
    for t in tokens:
        result.update(p for p in re.findall(r'[a-z]+|\d+', t) if len(p) >= 3)
    return result


def _filename_boost(query, filepath):
    fname_tokens  = _tokenize(os.path.splitext(os.path.basename(filepath))[0])
    query_tokens  = _tokenize(query)
    if not query_tokens:
        return 0.0
    coverage = len(query_tokens & fname_tokens) / len(query_tokens)
    if coverage >= 0.8: return 0.35
    if coverage >= 0.5: return 0.20
    if coverage >= 0.25: return 0.10
    return 0.0


def search_documents(query):
    if len(metadata) == 0:
        return []
    query_vec = model.encode(query)
    results   = []
    for i, doc_vec in enumerate(embeddings):
        score = cosine_similarity(query_vec, doc_vec) + \
                _filename_boost(query, metadata[i]["path"])
        results.append((score, metadata[i]["path"], "document"))
    results.sort(reverse=True)
    return results


def run_search(query, mode="both", session=None):
    if session is not None:
        session.refresh()
    results = []
    if mode in ("documents", "both"):
        results += search_documents(query)
    if mode in ("images", "both") and image_search_available:
        results += search_images(query)
    results.sort(reverse=True)
    if _auth_available and session is not None:
        log_search(session.username, query, mode, len(results))
    return results