import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

# -------- LOAD SBERT MODEL --------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
model = SentenceTransformer(MODEL_PATH, local_files_only=True)

# -------- LOAD DOCUMENT INDEX --------
def _load_doc_index():
    embed_path = "data/embeddings.npy"
    meta_path  = "data/documents.pkl"
    if not os.path.exists(embed_path) or not os.path.exists(meta_path):
        return np.empty((0, 384)), []
    emb = np.load(embed_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return emb, meta

embeddings, metadata = _load_doc_index()

# -------- LOAD IMAGE SEARCH --------
image_search_available = False
try:
    from image_search import search_images
    image_search_available = True
except ImportError:
    pass

# -------- LOAD AUDIT LOGGER --------
try:
    from auth import log_search
    _auth_available = True
except ImportError:
    _auth_available = False


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def reload_doc_index():
    global embeddings, metadata
    embeddings, metadata = _load_doc_index()


def search_documents(query: str):
    if len(metadata) == 0:
        return []
    query_vec = model.encode(query)
    raw = []
    for i, doc_vec in enumerate(embeddings):
        score = cosine_similarity(query_vec, doc_vec)
        raw.append((score, metadata[i]["path"]))
    raw.sort(reverse=True)
    seen = {}
    for score, path in raw:
        if path not in seen or score > seen[path]:
            seen[path] = score
    return sorted([(s, p, "document") for p, s in seen.items()], reverse=True)


def run_search(query: str, mode: str = "both", session=None):
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