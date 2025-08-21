# ========================
# RAG Retriever Setup
# ========================
import os, re, math, numpy as np, torch, chromadb
from datetime import datetime
from PIL import Image, ImageFile
from langchain_chroma import Chroma

# Optional inline display (Jupyter)
try:
    from IPython.display import display
except Exception:
    display = None

# LLM describe (safe fallback if unavailable/quota)
try:
    from src.llm import describe_image_with_metadata
except Exception:
    def describe_image_with_metadata(*args, **kwargs):
        return "(LLM off)"

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoid hangs on partial JPGs

# --- Reuse CLIP if already loaded ---
try:
    clip_model = clip_model  # noqa: F821
    clip_processor = clip_processor  # noqa: F821
except NameError:
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)

# Where the Chroma DB lives (env-first, default to local folder)
DB_DIR = os.getenv("DB_DIR", "vector_store")
client = chromadb.PersistentClient(path=DB_DIR)

# --- Consistent 512-d embedding wrapper ---
class CLIPEmbeddings:
    def __init__(self, clip_model, clip_processor):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
    def _embed_texts(self, texts):
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            embs = self.clip_model.get_text_features(**inputs).cpu().numpy()
        embs = embs / (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-12)
        return [e.tolist() for e in embs]
    def embed_query(self, text: str):
        return self._embed_texts([text])[0]
    def embed_documents(self, texts):
        return self._embed_texts(texts)

embedding_fn = CLIPEmbeddings(clip_model, clip_processor)

# --- Build retriever ---
vectorstore = Chroma(
    client=client,
    collection_name=os.getenv("COLLECTION_NAME", "satellite_images"),
    embedding_function=embedding_fn
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ========================
# Helpers (display + metadata utils)
# ========================
OPEN_IMAGES = os.getenv("OPEN_IMAGES", "0") == "1"

def _safe_open_and_display(path, max_px=1024, open_images: bool | None = None):
    """Show image in default viewer if open_images True; otherwise inline (if Jupyter).
       Resolves relative paths against CWD, script dir, and DB_DIR for robustness."""
    from os.path import abspath, isabs, exists, join, dirname
    should_open = OPEN_IMAGES if open_images is None else open_images

    # Resolve to an absolute path with fallbacks
    cand = [path]
    if not isabs(path):
        cwd = os.getcwd()
        script_dir = dirname(__file__)
        cand += [join(cwd, path), join(script_dir, path)]
        if os.getenv("DB_DIR"):
            cand.append(join(os.getenv("DB_DIR"), path))
    resolved = next((abspath(p) for p in cand if p and exists(p)), None)

    print(f"[preview] {path}")
    if not resolved:
        print(f" Could not open image (file not found in: {cand})")
        return

    try:
        if should_open:
            # Windows default viewer
            try:
                os.startfile(resolved)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
            # Cross-platform fallback
            from PIL import Image as _Img
            with _Img.open(resolved) as im:
                im = im.convert("RGB")
                im.thumbnail((max_px, max_px))
                im.show()
        else:
            if display:
                from PIL import Image as _Img
                with _Img.open(resolved) as im:
                    im = im.convert("RGB")
                    im.thumbnail((max_px, max_px))
                    display(im)
            else:
                print(f" (Inline display unavailable; set --open-images to open viewer)")
    except Exception as e:
        print(f" Could not open image {resolved}: {e}")


# tiny intent detector for ranking hints
_rx_high   = re.compile(r"\b(high|higher|highest|max|maximum|most)\b.*\bcloud", re.I)
_rx_low    = re.compile(r"\b(low|lower|lowest|min|minimal|least|clearest|fewest)\b.*\bcloud", re.I)
_rx_recent = re.compile(r"\b(recent|latest|newest|most\s*recent|last)\b", re.I)

def _cloud(v):
    try: return float(v)
    except: return math.nan

def _dt(v):
    try:
        # ISO-8601 like 2024-08-28T05:32:33.38Z
        return datetime.fromisoformat(str(v).replace("Z","+00:00"))
    except:
        return None

def _friendly_title(meta: dict, fallback_path: str | None):
    from os.path import basename, splitext
    return (
        meta.get("item_id") or
        meta.get("id") or
        meta.get("name") or
        meta.get("tile") or
        (splitext(basename(str(fallback_path)))[0] if fallback_path else None) or
        "image"
    )

def _all_items_from_collection(batch_size: int = 1000):
    """Fetch all docs from the underlying Chroma collection (metadatas, documents)."""
    try:
        coll = vectorstore._collection  # underlying chromadb Collection
        total = coll.count()
    except Exception as e:
        print("Could not access underlying collection:", e)
        return []
    out, off = [], 0
    while off < total:
        # FIX: 'ids' removed from include (Chroma >=0.5 disallows it)
        got = coll.get(include=["metadatas","documents"], limit=batch_size, offset=off)
        for _id, meta, doc in zip(got.get("ids",[]), got.get("metadatas",[]), got.get("documents",[])):
            class _Doc:
                def __init__(self, page_content, metadata):
                    self.page_content = page_content
                    self.metadata = metadata or {}
            out.append(_Doc(page_content=doc, metadata=meta or {}))
        off += batch_size
    return out

# ========================
# Unified Query Runners
# ========================
def run_query_dynamic(user_query: str, top_n=1, open_images: bool | None = None):
    """Your original dynamic runner: semantic search -> sort retrieved by lowest cloud."""
    print(f"\nUser Query: {user_query}")
    docs = retriever.invoke(user_query)
    if not docs:
        print("No relevant results found.")
        return

    docs_sorted = sorted(docs, key=lambda d: (d.metadata.get("cloud_cover")
                                              if d.metadata.get("cloud_cover") is not None else 999))
    doc = docs_sorted[0]
    meta = doc.metadata
    img_path = meta.get("preview_thumb") or meta.get("preview") or doc.page_content

    title = _friendly_title(meta, img_path)
    print(f"[retrieved] {title} — date: {meta.get('datetime')} — cloud: {meta.get('cloud_cover')}")
    if img_path and os.path.exists(img_path):
        _safe_open_and_display(img_path, max_px=1024, open_images=open_images)
        try:
            desc = describe_image_with_metadata(img_path, meta)
            print("Gemini:", desc)
        except Exception as e:
            print(f"Gemini describe failed: {e}")
    else:
        print(f"(Preview not found at {img_path})")

def run_query_vector_only(user_query: str, top_n: int = 2, describe: bool = True, k: int = 15, open_images: bool | None = None):
    """
    Pure vector search (CLIP + Chroma) -> pool of k
    BUT for cloud/recency queries we rank the ENTIRE collection by metadata (true min/max/latest).
    """
    print(f"\n[PURE RAG] User Query: {user_query}")

    want_high_cloud = bool(_rx_high.search(user_query))
    want_low_cloud  = bool(_rx_low.search(user_query))
    want_recent     = bool(_rx_recent.search(user_query))

    # For cloud/recency queries: use full collection
    if want_high_cloud or want_low_cloud or want_recent:
        docs = _all_items_from_collection()
    else:
        try:
            docs = retriever.vectorstore.similarity_search(user_query, k=k)
        except Exception:
            docs = retriever.invoke(user_query)

    if not docs:
        print("No relevant results found.")
        return

    ranked = docs
    if want_high_cloud or want_low_cloud:
        ranked = sorted(
            docs,
            key=lambda d: (_cloud(d.metadata.get("cloud_cover"))
                           if not math.isnan(_cloud(d.metadata.get("cloud_cover")))
                           else (-1e9 if want_low_cloud else 1e9)),
            reverse=want_high_cloud
        )
    elif want_recent:
        ranked = sorted(
            docs,
            key=lambda d: (_dt(d.metadata.get("datetime")) or datetime.min),
            reverse=True
        )

    hits = ranked[:top_n]
    for idx, d in enumerate(hits, 1):
        meta = d.metadata or {}
        img_path = meta.get("preview_thumb") or meta.get("preview") or d.page_content
        title = _friendly_title(meta, img_path)

        print(f"\nTop {idx}: {title} | date={meta.get('datetime')} | cloud={meta.get('cloud_cover')}")
        if img_path and os.path.exists(img_path):
            _safe_open_and_display(img_path, max_px=1024, open_images=open_images)
            if describe:
                try:
                    desc = describe_image_with_metadata(img_path, meta)
                    print("Gemini:", desc)
                except Exception as e:
                    print("Gemini describe failed:", e)
        else:
            print(f"(Preview not found at {img_path})")

# ========================
# CLI
# ========================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="GeoSpaider RAG queries")
    p.add_argument("queries", nargs="*", help="Queries to run (you can pass several)")
    p.add_argument("--mode", choices=["dynamic","pure"], default="pure", help="dynamic=retrieve then min-cloud; pure=vector (full coll for cloud/recent)")
    p.add_argument("--top-n", type=int, default=2, help="How many results to show for each query")
    p.add_argument("--k", type=int, default=15, help="Initial vector retrieval pool size (semantic mode)")
    p.add_argument("--no-describe", action="store_true", help="Disable LLM description step")
    p.add_argument("--open-images", action="store_true", help="Open images in viewer (also respects OPEN_IMAGES=1)")
    args = p.parse_args()

    if not args.queries:
        args.queries = [
            "show the most recent image of Bhopal",
            "show images with lowest cloud cover",
            "give me the highest cloud cover image available",
        ]

    for q in args.queries:
        if args.mode == "dynamic":
            run_query_dynamic(q, top_n=args.top_n, open_images=args.open_images)
        else:
            run_query_vector_only(
                q,
                top_n=args.top_n,
                describe=not args.no_describe,
                k=args.k,
                open_images=args.open_images,
            )
