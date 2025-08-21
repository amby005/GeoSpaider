# src/embed.py
import os
import pandas as pd
import torch
import numpy as np
from PIL import Image, ImageFile
from transformers import CLIPModel, CLIPProcessor
import chromadb

ImageFile.LOAD_TRUNCATED_IMAGES = True  # safer image loads

# -------- Config (relative to where you run the script) --------
META_CSV = os.getenv("META_CSV", "bhopal_s2/metadata.csv")
DB_DIR = os.getenv("DB_DIR", "vector_store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "satellite_images")

# -------- Helpers --------
def safe_norm(v: np.ndarray, eps: float = 1e-12):
    denom = np.linalg.norm(v, axis=-1, keepdims=True)
    denom = np.maximum(denom, eps)
    return v / denom

if __name__ == "__main__":
    print("[GeoSpaider] embed.py starting…")
    print(f" meta_csv={META_CSV}")
    print(f" db_dir={DB_DIR}")
    print(f" collection={COLLECTION_NAME}")

    # Load metadata
    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"metadata CSV not found: {META_CSV}")
    df = pd.read_csv(META_CSV)
    print(f"Loaded {len(df)} metadata entries")
    has_thumb = "preview_thumb" in df.columns

    # Init CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Init ChromaDB
    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f" Old collection '{COLLECTION_NAME}' deleted (reset).")
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    # Embed & store
    added = 0
    for i, row in df.iterrows():
        # choose preview path (thumb > preview)
        img_path = row.get("preview_thumb") if has_thumb else row.get("preview")
        if isinstance(img_path, str):
            img_path = img_path.replace("\\", "/")  # normalize POSIX paths

        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f" Missing/invalid image path: {img_path}, skipping.")
            continue

        # Image embedding
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                img_inputs = processor(images=im, return_tensors="pt").to(device)
        except Exception as e:
            print(f" Failed to open/process image {img_path}: {e}. Skipping.")
            continue
        with torch.no_grad():
            emb_img = model.get_image_features(**img_inputs).cpu().numpy()

        # Text embedding (row metadata as text)
        meta_text = str(row.to_dict())
        text_inputs = processor(text=[meta_text], return_tensors="pt",
                                padding=True, truncation=True).to(device)
        with torch.no_grad():
            emb_txt = model.get_text_features(**text_inputs).cpu().numpy()

        # Fuse (normalized average)
        emb_img = safe_norm(emb_img)
        emb_txt = safe_norm(emb_txt)
        combined = safe_norm((emb_img + emb_txt) / 2.0)[0].tolist()

        # Clean/normalize metadata we store in Chroma
        meta = row.to_dict()
        # normalize preview fields (always include preview; mirror thumb if present)
        meta["preview"] = (row.get("preview") or "").replace("\\", "/")
        if has_thumb:
            meta["preview_thumb"] = (row.get("preview_thumb") or "").replace("\\", "/")
        # guarantee an identifier field
        meta["item_id"] = str(row.get("item_id") or row.get("id") or f"id_{i}")
        meta["id"] = meta["item_id"]  # mirror for convenience

        # Store in Chroma
        collection.add(
            ids=[meta["item_id"]],
            embeddings=[combined],
            metadatas=[meta],
            documents=[img_path]
        )
        added += 1

    print(f"Stored {added} multimodal (img+text) embeddings in ChromaDB at {DB_DIR}")
