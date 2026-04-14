from sentence_transformers import SentenceTransformer
import os
# Tell the Hugging Face transformers library to ONLY use PyTorch and ignore TensorFlow
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_TIMEOUT"] = "120"   # increase more (very important)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print("Downloading and saving model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
save_path = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
model.save(save_path)
print(f"Model saved to: {save_path}")
