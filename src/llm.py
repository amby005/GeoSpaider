from dotenv import load_dotenv
load_dotenv()
import os

def gemini_available() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY"))

def describe_image_with_metadata(img_path: str, metadata: dict) -> str:
    """Return a short description using Gemini if key is set; otherwise a fallback."""
    if not gemini_available():
        dt = metadata.get("datetime"); cc = metadata.get("cloud_cover")
        return f"(LLM off) {metadata.get('item_id','image')} — date={dt}, cloud={cc}."
    import google.generativeai as genai
    from PIL import Image as PILImage
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    image = PILImage.open(img_path).convert("RGB")
    prompt = f"""
    You are an expert satellite imagery analyst.
    Metadata: {metadata}
    Generate a short, precise description of what is visible.
    Mention date and cloud cover explicitly and note any cloud limitations.
    """
    resp = model.generate_content([prompt, image])
    return resp.text
