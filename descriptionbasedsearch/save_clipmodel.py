from sentence_transformers import SentenceTransformer
import os
# Tell the Hugging Face transformers library to ONLY use PyTorch and ignore TensorFlow
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_TIMEOUT"] = "120"   # increase more (very important)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print("Saving CLIP model...")
clipmodel = SentenceTransformer("clip-ViT-B-32")
save_path2=os.path.join(os.path.dirname(__file__), "models", "clip-ViT-B-32")
clipmodel.save(save_path2)
print(f"Model saved to: {save_path2}")