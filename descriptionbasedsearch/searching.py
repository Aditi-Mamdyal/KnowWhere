import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

# Load model and data
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.load("data/embeddings.npy")

with open("data/documents.pkl", "rb") as f:
    metadata = pickle.load(f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

query = input("Enter search query: ")
query_vec = model.encode(query)

scores = []
for i, doc_vec in enumerate(embeddings):
    score = cosine_similarity(query_vec, doc_vec)
    scores.append((score, metadata[i]))

# Sort by similarity
scores.sort(reverse=True)

print("\nTop results:\n")
for score, path in scores[:3]:
    print(f"{score:.3f} → {path}")
