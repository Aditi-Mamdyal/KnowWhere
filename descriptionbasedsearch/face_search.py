"""
face_search.py
==============
Improved face search with multi-reference voting for better accuracy.

KEY IMPROVEMENTS:
  1. Uses ALL photos in face_database/PersonName/ instead of just the first one.
  2. An image must match MIN_REFERENCE_VOTES references to be counted.
  3. Confidence is averaged across all matching references.
  4. Slightly relaxed threshold (0.45) to catch ID cards, angled shots, etc.
  5. Deduplication — same image never appears twice.

HOW TO IMPROVE ACCURACY:
  Add more varied reference photos to face_database/PersonName/:
    - Front facing, clear lighting
    - Slight angle (left/right)
    - With glasses if the person wears them
    - Different backgrounds/lighting
  3–6 photos is the sweet spot.
"""

import os
import shutil
import logging
from deepface import DeepFace

logger = logging.getLogger("FaceSearch")
logger.setLevel(logging.WARNING)

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = r"D:\coding\college"
FACE_DB_PATH = os.path.join(BASE_DIR, "face_database")

# ── TUNABLE PARAMETERS ────────────────────────────────────────────────────────
# How many reference photos must match for a candidate to count.
# Start at 1. If you see wrong people, increase to 2.
MIN_REFERENCE_VOTES = 1

# Distance threshold — lower = stricter. Default VGG-Face is 0.40.
# 0.45 is slightly more lenient to catch ID cards and angled photos.
# If too many wrong people appear, lower to 0.35.
DISTANCE_THRESHOLD = 0.45

# Max reference photos per person. More = more accurate but slower.
# 6 is a good balance.
MAX_REFERENCES = 6

VALID_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# ── INTERNAL HELPERS ──────────────────────────────────────────────────────────

def _get_reference_images(person_name: str) -> list:
    """Returns list of reference image paths for a person, capped at MAX_REFERENCES."""
    person_folder = os.path.join(FACE_DB_PATH, person_name)
    if not os.path.exists(person_folder):
        logger.warning(f"No folder found for '{person_name}' in face_database.")
        return []
    images = [
        os.path.join(person_folder, f)
        for f in os.listdir(person_folder)
        if f.lower().endswith(VALID_EXTS)
    ]
    if not images:
        logger.warning(f"Folder for '{person_name}' has no valid images.")
        return []
    return images[:MAX_REFERENCES]


def _load_image_as_rgb(filepath: str):
    """
    Load image with PIL and convert to RGB numpy array.
    Handles RGBA, palette mode, screenshots, and unusual formats
    that cause DeepFace's 'list has no attribute ndim' error.
    Returns numpy array or None on failure.
    """
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(filepath)
        # Convert any mode to RGB (handles RGBA, P, L, CMYK etc.)
        img = img.convert("RGB")
        return np.array(img)
    except Exception as e:
        logger.error(f"Cannot load image '{os.path.basename(filepath)}': {e}")
        return None


def _verify_one(reference_path: str, candidate_path: str) -> tuple:
    """
    Runs DeepFace.verify between one reference and one candidate.
    Pre-processes both images with PIL to avoid the
    'list object has no attribute ndim' error caused by
    RGBA/alpha channel images and unusual formats like screenshots.
    Returns (matched: bool, confidence: float).
    """
    # Pre-load both images as RGB numpy arrays
    ref_img = _load_image_as_rgb(reference_path)
    if ref_img is None:
        return False, 0.0

    can_img = _load_image_as_rgb(candidate_path)
    if can_img is None:
        return False, 0.0

    try:
        # Pass numpy arrays directly — avoids DeepFace re-reading the file
        # which is where the format errors occur
        result = DeepFace.verify(
            img1_path=ref_img,
            img2_path=can_img,
            enforce_detection=False,
            detector_backend="opencv",
            model_name="VGG-Face",
            distance_metric="cosine",
            silent=True
        )
        distance   = result["distance"]
        matched    = distance <= DISTANCE_THRESHOLD
        confidence = max(0.0, 1.0 - distance)
        return matched, confidence
    except Exception as e:
        logger.error(f"DeepFace error: {os.path.basename(reference_path)} "
                     f"vs {os.path.basename(candidate_path)}: {e}")
        return False, 0.0


# ── MAIN SEARCH FUNCTION ──────────────────────────────────────────────────────

def search_face(query_name: str) -> list:
    """
    Search for images containing a specific person using multi-reference voting.

    Args:
        query_name: folder name inside face_database/ (e.g. "Aditi", "Suraj")

    Returns:
        List of (confidence, image_path) sorted best first.
        confidence is between 0.0 and 1.0.
    """
    # create face_database if it doesn't exist
    if not os.path.exists(FACE_DB_PATH):
        logger.error(f"face_database folder not found: {FACE_DB_PATH}")
        os.makedirs(FACE_DB_PATH, exist_ok=True)
        return []

    references = _get_reference_images(query_name)
    if not references:
        return []

    print(f"[FACE] Searching for '{query_name}' "
          f"using {len(references)} reference photo(s).")

    # candidate_scores maps image_path -> list of confidence scores from each matching reference
    candidate_scores = {}

    for root, _, files in os.walk(IMAGE_FOLDER):
        for file in files:
            if not file.lower().endswith(VALID_EXTS):
                continue

            candidate_path = os.path.join(root, file)

            # skip the reference photos themselves
            if candidate_path in references:
                continue

            # compare candidate against every reference photo
            match_confidences = []
            for ref_path in references:
                matched, conf = _verify_one(ref_path, candidate_path)
                if matched:
                    match_confidences.append(conf)

            # only count if it matched enough references
            if len(match_confidences) >= MIN_REFERENCE_VOTES:
                avg_confidence = sum(match_confidences) / len(match_confidences)
                candidate_scores[candidate_path] = avg_confidence

    results = sorted(
        [(conf, path) for path, conf in candidate_scores.items()],
        reverse=True
    )

    print(f"[FACE] Found {len(results)} image(s) matching '{query_name}'.")
    return results


# ── UTILITY FUNCTIONS (used by GUI admin panel) ───────────────────────────────

def add_reference_photo(person_name: str, photo_path: str) -> dict:
    """
    Copies a photo into face_database/PersonName/.
    Called from the Admin Panel Face Registration tab.

    Returns:
        {"success": True, "saved_as": path}  or
        {"success": False, "reason": "..."}
    """
    person_folder = os.path.join(FACE_DB_PATH, person_name)
    os.makedirs(person_folder, exist_ok=True)

    if not os.path.isfile(photo_path):
        return {"success": False, "reason": f"Photo not found: {photo_path}"}

    ext = os.path.splitext(photo_path)[1].lower()
    if ext not in VALID_EXTS:
        return {"success": False,
                "reason": f"Unsupported format '{ext}'. Use jpg, png, jpeg, webp, or bmp."}

    existing = [f for f in os.listdir(person_folder)
                if f.lower().endswith(VALID_EXTS)]
    new_name = f"ref_{len(existing) + 1}{ext}"
    dest     = os.path.join(person_folder, new_name)

    try:
        shutil.copy2(photo_path, dest)
        return {"success": True, "saved_as": dest}
    except Exception as e:
        return {"success": False, "reason": str(e)}


def list_registered_people() -> list:
    """
    Returns list of people registered in face_database/.
    Used by GUI to populate the Face Registration table.
    """
    if not os.path.exists(FACE_DB_PATH):
        return []
    return [
        name for name in os.listdir(FACE_DB_PATH)
        if os.path.isdir(os.path.join(FACE_DB_PATH, name))
    ]


# ── STANDALONE TEST ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    people = list_registered_people()
    if not people:
        print("No people registered in face_database/")
        print("Create folders like: face_database/Aditi/ and add reference photos.")
    else:
        print(f"Registered people: {people}")
        name    = input("Search for (enter name exactly): ").strip()
        results = search_face(name)
        if not results:
            print("No matches found.")
        else:
            print(f"\nTop matches for '{name}':")
            for conf, path in results[:5]:
                print(f"  {conf:.2%} confidence → {path}")