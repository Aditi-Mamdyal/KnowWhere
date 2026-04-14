import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from face_search import search_face

os.environ["USE_TF"] = "0"

# -------- CONFIG --------
DATA_DIR   = "data"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "clip-ViT-B-32")
EMBED_PATH = os.path.join(DATA_DIR, "clip_embeddings.npy")
META_PATH  = os.path.join(DATA_DIR, "clip_documents.pkl")

# -------- LOAD MODEL ONCE AT STARTUP --------
print("[IMAGE SEARCH] Loading CLIP model...")
model = SentenceTransformer(MODEL_PATH, local_files_only=True)

# -------- LOAD INDEX ONCE AT STARTUP --------
# Previously load_index() was called inside search_images() which reloaded
# the entire index from disk on every single search. Fixed — load once here.
def _load_index():
    if not os.path.exists(EMBED_PATH) or not os.path.exists(META_PATH):
        return np.array([]), []
    emb = np.load(EMBED_PATH, allow_pickle=True)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return emb, meta

embeddings, metadata = _load_index()


def reload_index():
    """
    Call this after image_indexing.py runs to refresh the in-memory index.
    indexing_service.py calls this after index_images() completes.
    """
    global embeddings, metadata
    embeddings, metadata = _load_index()


def search_images(query, top_k=5):
    if len(metadata) == 0:
        print("[WARN] No images indexed.")
        return []

    query_lower      = query.lower()
    face_db_path     = "face_database"
    detected_person  = None

    # Step 1: detect person name in query
    if os.path.exists(face_db_path):
        for person in os.listdir(face_db_path):
            if person.lower() in query_lower:
                detected_person = person
                break

    # Step 2: encode query with CLIP
    query_vec = model.encode(query)

    if detected_person:
        # Hybrid: CLIP scores filtered by face recognition matches
        print(f"[HYBRID] Searching for faces matching: {detected_person}")
        face_results = search_face(detected_person)
        face_paths   = {path for _, path in face_results}

        hybrid_results = []
        for i, m in enumerate(metadata):
            if m["path"] in face_paths:
                img_vec = embeddings[i]
                sim     = np.dot(query_vec, img_vec) / (norm(query_vec) * norm(img_vec))
                hybrid_results.append((sim, m["path"], "image"))

        hybrid_results.sort(key=lambda x: x[0], reverse=True)

        # fallback — if CLIP+face found nothing, return face-only results
        if not hybrid_results:
            print("[HYBRID] No combined match — showing face results only")
            return [(1.0, p, "image") for _, p in face_results[:top_k]]

        return hybrid_results[:top_k]

    else:
        # Standard CLIP description search
        print("[CLIP] Searching by description...")
        scores = []
        for i, img_vec in enumerate(embeddings):
            sim = np.dot(query_vec, img_vec) / (norm(query_vec) * norm(img_vec))
            scores.append((sim, metadata[i]["path"], "image"))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]