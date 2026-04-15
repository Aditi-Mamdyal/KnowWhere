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

# -------- LOAD AUDIT LOGGER --------
try:
    from auth import log_search
    auth_available = True
except ImportError:
    auth_available = False


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


def run_search(query: str, mode: str = "both", session=None):
    """
    Unified search entry point for GUI and CLI.

    query:   what the user typed
    mode:    "documents" / "images" / "both"
    session: optional Session object from auth.py — used for audit logging
             and session refresh on every search action
    """
    # Refresh session on every search so timeout resets
    if session is not None:
        session.refresh()

    results = []

    if mode in ("documents", "both"):
        results += search_documents(query)

    if mode in ("images", "both"):
        if image_search_available:
            results += search_images(query)
        elif mode == "images":
            print("[WARN] Image search not available.")

    # Sort all results together by score
    results.sort(reverse=True)

    # Log to audit trail if auth is available and session provided
    if auth_available and session is not None:
        log_search(
            username=session.username,
            query=query,
            mode=mode,
            result_count=len(results)
        )

    return results


def display_results(results, top_n=3):
    """Print results with confidence labels."""
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


# -------- CLI MODE --------
if __name__ == "__main__":
    from auth import verify_login, Session, log_logout

    # Simple CLI login
    print("=" * 40)
    print("  Corporate Document Search System")
    print("=" * 40)

    username = input("\nUsername: ").strip()
    password = input("Password: ").strip()

    result = verify_login(username, password)

    if not result["success"]:
        print(f"\nLogin failed: {result['reason']}")
        exit(1)

    session = Session(result["username"], result["role"])
    print(f"\nWelcome, {session.username} ({session.role})")
    print("Type 'quit' to exit.\n")

    while True:
        if not session.is_valid():
            print("\nSession expired. Please log in again.")
            break

        query = input("Search query: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        mode = input("Mode [documents/images/both] (default: both): ").strip().lower()
        if mode not in ("documents", "images", "both"):
            mode = "both"

        results = run_search(query, mode=mode, session=session)
        display_results(results)
        print()

    log_logout(session.username)
    print("Logged out.")