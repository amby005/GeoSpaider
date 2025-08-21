import os, json, requests, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pystac_client import Client
import rioxarray
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # allow large sat images safely

class MCPInterface:
    def __init__(self):
        self.client = Client.open("https://earth-search.aws.element84.com/v1")

    def search_satellite_images(self, bbox, date_range, limit, max_cloud=None):
        query = {"eo:cloud_cover": {"lt": max_cloud}} if max_cloud is not None else None
        search = self.client.search(
            collections=["sentinel-2-l2a"],
            datetime=date_range,
            bbox=bbox,
            limit=limit,
            query=query
        )
        return list(search.get_items())

    def fetch_and_save(self, items, out_dir):
        results = []
        for i, item in enumerate(items, 1):
            print(f"\n[{i}/{len(items)}] {item.id}")
            result = download_rgb_or_visual(item, os.path.join(out_dir, "downloads"))
            if result is None:
                print(" No RGB/visual available — skipping.")
                continue

            meta = collect_metadata(item)
            os.makedirs(os.path.join(out_dir, "previews"), exist_ok=True)

            if result["type"] == "bands":
                jpg_path = os.path.join(out_dir, "previews", f"{item.id}.jpg")
                build_truecolor_jpeg(result["files"], jpg_path)
                shrink_image_in_place(jpg_path, max_px=1024, quality=85)
                meta["preview"] = jpg_path
                meta.update(result["files"])
                print(f" Saved reduced RGB preview {jpg_path}")
            else:
                jpg_path = result["files"]["jpg"]
                shrink_image_in_place(jpg_path, max_px=1024, quality=85)
                new_path = os.path.join(out_dir, "previews", f"{item.id}.jpg")
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                if jpg_path != new_path:
                    try:
                        os.replace(jpg_path, new_path)
                        jpg_path = new_path
                    except Exception:
                        pass
                meta["preview"] = jpg_path
                print(f" Saved reduced visual preview {jpg_path}")

            results.append(meta)
        return results

# ---- Config (edit paths if you want output elsewhere)
BBOX = [77.35, 23.20, 77.45, 23.30]
DATE_RANGE = "2024-06-01/2024-08-31"
LIMIT = 5
OUT_DIR = "bhopal_s2"   # this will be created under the current working directory

# ---- Helpers
def safe_download(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(1024*1024):
            if chunk:
                f.write(chunk)
    return out_path

def download_rgb_bands(item, out_root):
    band_map = {"red": "B04", "green": "B03", "blue": "B02"}
    band_files = {}
    for color, asset_key in band_map.items():
        if asset_key not in item.assets:
            print(f" Band {asset_key} missing in {item.id}")
            return None
        url = item.assets[asset_key].href
        out_path = os.path.join(out_root, f"{item.id}_{color}.tif")
        safe_download(url, out_path)
        band_files[color] = out_path
    return band_files

def build_truecolor_jpeg(band_files, out_jpg):
    R = rioxarray.open_rasterio(band_files["red"]).squeeze().astype(np.float32)
    G = rioxarray.open_rasterio(band_files["green"]).squeeze().astype(np.float32)
    B = rioxarray.open_rasterio(band_files["blue"]).squeeze().astype(np.float32)
    R, G, B = R[::10, ::10], G[::10, ::10], B[::10, ::10]

    def norm(x):
        x = x.values if hasattr(x, "values") else x
        mn, mx = np.nanpercentile(x, 2), np.nanpercentile(x, 98)
        return np.clip((x - mn) / (mx - mn + 1e-6), 0, 1)

    rgb = np.dstack([norm(R), norm(G), norm(B)])
    plt.figure(figsize=(6,6))
    plt.imshow(rgb)
    plt.axis("off")
    os.makedirs(os.path.dirname(out_jpg), exist_ok=True)
    plt.savefig(out_jpg, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()
    return out_jpg

def shrink_image_in_place(in_path, max_px=1024, quality=85):
    with Image.open(in_path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_px, max_px))
        im.save(in_path, "JPEG", optimize=True, quality=quality)

def download_rgb_or_visual(item, out_root):
    if all(k in item.assets for k in ["B04", "B03", "B02"]):
        band_files = download_rgb_bands(item, out_root)
        if band_files:
            return {"type": "bands", "files": band_files}
    if "visual" in item.assets:
        url = item.assets["visual"].href
        out_path = os.path.join(out_root, f"{item.id}_visual.jpg")
        safe_download(url, out_path)
        return {"type": "visual", "files": {"jpg": out_path}}
    return None

def collect_metadata(item):
    props = item.properties
    return {
        "item_id": item.id,
        "datetime": props.get("datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "platform": props.get("platform"),
        "constellation": props.get("constellation"),
        "instruments": props.get("instruments"),
        "mgrs_tile": props.get("s2:mgrs_tile"),
    }

# ---- Script entry
if __name__ == "__main__":
    os.makedirs(os.path.join(OUT_DIR, "downloads"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "previews"), exist_ok=True)

    mcp = MCPInterface()
    items = mcp.search_satellite_images(BBOX, DATE_RANGE, LIMIT)
    print(f" Found {len(items)} Sentinel-2 items")

    records = mcp.fetch_and_save(items, OUT_DIR)

    import pandas as pd, json
    pd.DataFrame(records).to_csv(os.path.join(OUT_DIR, "metadata.csv"), index=False)
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(records, f, indent=2)

    print("\nDone. Reduced-size previews saved in", OUT_DIR)
