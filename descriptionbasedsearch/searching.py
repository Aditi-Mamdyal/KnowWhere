import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

# -------- LOAD SBERT MODEL --------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
model = SentenceTransformer(MODEL_PATH, local_files_only=True)

# -------- LOAD DOCUMENT INDEX --------
embeddings = np.load("data/embeddings.npy")
with open("data/documents.pkl", "rb") as f:
    metadata = pickle.load(f)

# -------- LOAD IMAGE SEARCH (only if available) --------
image_search_available = False
try:
    from image_search import search_images
    image_search_available = True
except ImportError:
    pass


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def search_documents(query: str):
    """Search text documents using SBERT. Returns list of (score, path, type)."""
    query_vec = model.encode(query)
    scores    = []

    for i, doc_vec in enumerate(embeddings):
        score = cosine_similarity(query_vec, doc_vec)
        scores.append((score, metadata[i]["path"]))

    scores.sort(reverse=True)

    # Deduplicate — chunking means same file appears multiple times.
    # Keep only the highest scoring chunk per file.
    seen = {}
    for score, path in scores:
        if path not in seen or score > seen[path]:
            seen[path] = score

    return sorted(
        [(score, path, "document") for path, score in seen.items()],
        reverse=True
    )


def run_search(query: str, mode: str = "both"):
    """
    Unified search entry point for GUI and CLI.

    mode: "documents" — text documents only (SBERT)
          "images"    — images only (CLIP + face)
          "both"      — everything merged and ranked together
    """
    results = []

    if mode in ("documents", "both"):
        results += search_documents(query)

    if mode in ("images", "both"):
        if image_search_available:
            results += search_images(query)
        elif mode == "images":
            print("[WARN] Image search not available — check image_search.py is present.")

    # Sort all results together by score descending
    results.sort(reverse=True)
    return results


def display_results(results, top_n=3):
    """Print results with confidence labels. Used by CLI and can be used by GUI."""
    if not results:
        print("No results found.")
        return

    if results[0][0] < 0.3:
        print("\nNo strong matches found. Showing closest results anyway.")

    print("\nTop results:\n")
    for score, path, result_type in results[:top_n]:
        if score >= 0.5:
            confidence = "Strong"
        elif score >= 0.35:
            confidence = "Moderate"
        elif score >= 0.25:
            confidence = "Weak"
        else:
            confidence = "Very weak"
        print(f"  [{confidence}][{result_type}] {score:.3f} → {path}")


# -------- RUN DIRECTLY (CLI mode) --------
if __name__ == "__main__":
    query   = input("Enter search query: ")
    mode    = input("Search [documents / images / both]: ").strip().lower()
    if mode not in ("documents", "images", "both"):
        mode = "both"

    results = run_search(query, mode=mode)
    display_results(results)