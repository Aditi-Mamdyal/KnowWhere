import os
import pickle
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import logging

# Set the environment variable to avoid the Keras 3 / Transformers conflict
os.environ["USE_TF"] = "0"
# ── ROOT: absolute path so launch.bat works correctly ─────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))

# CONFIG - Aligned with your computer
DOC_FOLDER     = r"D:\coding\college"
DATA_DIR   = os.path.join(_ROOT, "data")
EMBED_PATH     = os.path.join(DATA_DIR, "clip_embeddings.npy")
META_PATH      = os.path.join(DATA_DIR, "clip_documents.pkl")
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "models", "clip-ViT-B-32")

os.makedirs(DATA_DIR, exist_ok=True)
logging.getLogger("PIL").setLevel(logging.ERROR)

# Load CLIP locally
print("[IMAGE INDEX] Loading local CLIP model...")
model = SentenceTransformer(MODEL_PATH, local_files_only=True)

def index_images():
    # Load existing data to support incremental indexing
    embeddings = np.array([])
    metadata = []
    
    if os.path.exists(EMBED_PATH) and os.path.exists(META_PATH):
        try:
            embeddings = np.load(EMBED_PATH)
            with open(META_PATH, "rb") as f:
                metadata = pickle.load(f)
        except Exception:
            print("[WARN] Image index corrupted. Rebuilding...")

    # Map paths to mtimes for quick comparison
    indexed_files = {m['path']: m['mtime'] for m in metadata}
    
    new_vectors = []
    new_metadata = []
    current_files_on_disk = set()

    print(f"[IMAGE INDEX] Scanning {DOC_FOLDER}...")

    for root, _, files in os.walk(DOC_FOLDER):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                continue
                
            full_path = os.path.join(root, file)
            current_files_on_disk.add(full_path)
            
            try:
                mtime = os.path.getmtime(full_path)
                
                # INCREMENTAL CHECK: Skip if file exists and hasn't changed
                if full_path in indexed_files and mtime <= indexed_files[full_path]:
                    continue

                # Process new/updated image
                img = Image.open(full_path).convert("RGB")
                vec = model.encode(img)
                
                new_vectors.append(vec)
                new_metadata.append({"path": full_path, "mtime": mtime})
                print(f"  [NEW] Indexed: {file}")

            except Exception as e:
                print(f"  [ERROR] Skipping {file}: {e}")

    # CLEANUP: Remove files from index that were deleted from disk
    deleted_indices = [i for i, m in enumerate(metadata) if m["path"] not in current_files_on_disk]
    if deleted_indices:
        print(f"[CLEANUP] Removing {len(deleted_indices)} missing images from index.")
        if embeddings.size > 0:
            embeddings = np.delete(embeddings, deleted_indices, axis=0)
        metadata = [m for i, m in enumerate(metadata) if i not in deleted_indices]

    # MERGE & SAVE
    if new_vectors:
        new_arr = np.array(new_vectors)
        embeddings = new_arr if embeddings.size == 0 else np.vstack((embeddings, new_arr))
        metadata.extend(new_metadata)

    np.save(EMBED_PATH, embeddings)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"[SUCCESS] Image Index Size: {len(metadata)}")

if __name__ == "__main__":
    index_images()