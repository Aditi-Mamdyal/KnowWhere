import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")
print("loaded")

DOC_FOLDER = r"D:\coding\college"
DATA_DIR = "data"

EMBED_PATH= os.path.join(DATA_DIR, "embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "documents.pkl")

texts = []
metadata = []

def extract_text(filepath):
    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if filepath.endswith(".pdf"):
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    if filepath.endswith(".docx"):
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])

    return ""

# Scan documents folder
for root, _, files in os.walk(DOC_FOLDER):
    for file in files:
        path = os.path.join(root, file)
        text = extract_text(path)

        if text.strip():
            texts.append(text)
            metadata.append(path)
print(metadata)
# Create embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Save to disk
np.save(EMBED_PATH, embeddings)
with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("Indexing completed.")