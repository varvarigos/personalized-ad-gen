import base64
import io
import json
from typing import List, Optional, Tuple

import gradio as gr
from PIL import Image
from google import genai

COOKIE_NAME = "shop_profile"


# ---------- Cookie helpers (REAL site cookie, no JWT) ----------
def _empty_profile() -> dict:
    return {
        "search_history": [],  # most-recent-first
        "interests": [],  # deduped tags inferred from searches
    }


def _loads_cookie(val: Optional[str]) -> dict:
    if not val:
        return _empty_profile()
    try:
        data = base64.urlsafe_b64decode(
            val.encode("utf-8") + b"==="
        )  # padding tolerant
        obj = json.loads(data.decode("utf-8"))
        if not isinstance(obj, dict):
            return _empty_profile()
        obj.setdefault("search_history", [])
        obj.setdefault("interests", [])
        if not isinstance(obj["search_history"], list):
            obj["search_history"] = []
        if not isinstance(obj["interests"], list):
            obj["interests"] = []
        return obj
    except Exception:
        return _empty_profile()


def _dumps_cookie(obj: dict) -> str:
    raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _merge_profiles(p_req: dict, p_shadow: dict) -> dict:
    base = (
        p_req
        if len(p_req.get("search_history", []))
        >= len(p_shadow.get("search_history", []))
        else p_shadow
    )
    other = p_shadow if base is p_req else p_req

    hist = base.get("search_history", []).copy()
    for q in other.get("search_history", []):
        if q not in hist:
            hist.append(q)
    hist = hist[:20]

    intr = base.get("interests", []).copy()
    seen = {i.lower() for i in intr}
    for i in other.get("interests", []):
        if i and i.lower() not in seen:
            intr.append(i)
            seen.add(i.lower())
    intr = intr[:12]

    return {"search_history": hist, "interests": intr}


# ---------- Prompt templates ----------
def build_ad_prompt(profile: dict, latest_query: str) -> str:
    recency = []
    seen = set()
    for q in [latest_query] + profile.get("search_history", []):
        k = (q or "").strip()
        if k and k.lower() not in seen:
            recency.append(k)
            seen.add(k.lower())
    recency = recency[:6]
    history = profile.get("search_history", [])[:6]

    return (
        "Generate a high-quality, brand-agnostic e-commerce advertisement hero image.\n"
        f"Primary product focus (highest priority): {latest_query}.\n"
        f"Recent related interests (ordered by recency): {', '.join(recency)}.\n"
        f"Recent browsing themes: {', '.join(history)}.\n"
        "Design guidelines:\n"
        "• Photorealistic product presentation with studio-quality lighting.\n"
        "• Clean, modern environment with subtle contextual props that match the product (e.g., turf for sports gear, desk for office items).\n"
        "• Neutral/light background to ensure overlay readability.\n"
        "• Natural shadows and gentle depth-of-field for realism.\n"
        "• Leave clear negative space for a headline and a 'Shop Now' CTA button.\n"
        "• Aspect ratio 4:3; composition suitable for web store hero banners.\n"
        "• Encourage brand logos and text."
    )


def build_continuation_prompt(profile: dict, focus_query: str) -> str:
    recency = []
    seen = set()
    for q in [focus_query] + profile.get("search_history", []):
        k = (q or "").strip()
        if k and k.lower() not in seen:
            recency.append(k)
            seen.add(k.lower())
    recency = recency[:6]
    history = profile.get("search_history", [])[:6]

    return (
        "Create a follow-up e-commerce advertisement hero image that feels like a natural continuation of the previous creative.\n"
        f"Maintain visual continuity but introduce a new, complementary scene or arrangement.\n"
        f"Primary product focus: {focus_query}.\n"
        f"Related interests (recency first): {', '.join(recency)}.\n"
        f"Recent browsing themes: {', '.join(history)}.\n"
        "Design guidelines:\n"
        "• Keep lighting, palette, and realism consistent with the prior image.\n"
        "• Add complementary elements to evolve the story (e.g., if football was shown, now include a goal post or pitch lines).\n"
        "• Preserve clean, neutral background suitable for overlays; leave space for headline + 'Shop Now' CTA.\n"
        "• Subtle change in camera angle or composition (still 4:3) to feel fresh without breaking continuity.\n"
        "• Encourage brand logos and text."
    )


# ---------- Model helpers ----------
def generate_image_from_prompt(
    prompt: str,
) -> Tuple[Optional[Image.Image], Optional[str]]:
    try:
        client = genai.Client(api_key="AIzaSyDKVznwVpIG0u4Kf3TjtxZDRRlf0qS_o-A")
        resp = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=prompt,
        )
        image_bytes = None
        if getattr(resp, "candidates", None):
            parts = getattr(resp.candidates[0].content, "parts", []) or []
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    image_bytes = inline.data
                    break
        if not image_bytes:
            return None, "Model returned no inline image data. Try another query."
        img = Image.open(io.BytesIO(image_bytes))
        return img, None
    except Exception as e:
        return None, f"Error: {e}"


def _pil_to_inline_part(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/png", "data": data_b64}}


def _gallery_item_to_pil(item):
    """Normalize a gallery item (tuple/path/URL/PIL/numpy/dict) into a PIL.Image or None."""
    if item is None:
        return None
    if isinstance(item, (tuple, list)) and item:
        return _gallery_item_to_pil(item[0])
    if isinstance(item, dict):
        return _gallery_item_to_pil(item.get("image") or item.get("value"))
    if isinstance(item, Image.Image):
        return item
    try:
        import numpy as np

        if isinstance(item, np.ndarray):
            return Image.fromarray(item)
    except Exception:
        pass
    if isinstance(item, str):
        try:
            return Image.open(item)
        except Exception:
            return None
    return None


def generate_continuation_image(
    prompt: str, prev_img: Optional[Image.Image]
) -> Tuple[Optional[Image.Image], Optional[str]]:
    try:
        contents = [prompt]
        if prev_img is not None:
            contents.append(_pil_to_inline_part(prev_img))

        client = genai.Client(api_key="AIzaSyDKVznwVpIG0u4Kf3TjtxZDRRlf0qS_o-A")
        resp = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=contents,
        )
        image_bytes = None
        if getattr(resp, "candidates", None):
            parts = getattr(resp.candidates[0].content, "parts", []) or []
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    image_bytes = inline.data
                    break
        if not image_bytes:
            return None, "Model returned no inline image data on continuation."
        img = Image.open(io.BytesIO(image_bytes))
        return img, None
    except Exception as e:
        return None, f" Error: {e}"


# ---------- Search / Rotate handlers ----------
def handle_search(
    query: str,
    cookie_shadow: str,
    gallery: List[Image.Image],
    search_counter: int,
    request: gr.Request,
):
    q = (query or "").strip()
    if not q:
        return (
            cookie_shadow,
            "Please type a search term.",
            None,
            gallery,
            "No search yet.",
            search_counter,
        )

    # Merge profiles from request + shadow
    req_cookie_val = request.cookies.get(COOKIE_NAME)
    prof_req = _loads_cookie(req_cookie_val)
    prof_shadow = _loads_cookie(cookie_shadow)
    profile = _merge_profiles(prof_req, prof_shadow)

    # Update with this search (recency first)
    hist = profile.setdefault("search_history", [])
    if q in hist:
        hist.remove(q)
    hist.insert(0, q)
    del hist[20:]

    intr = profile.setdefault("interests", [])
    low = [i.lower() for i in intr]
    if q.lower() in low:
        idx = low.index(q.lower())
        intr.pop(idx)
    intr.insert(0, q)
    del intr[12:]

    # Build prompt and generate image
    ad_prompt = build_ad_prompt(profile, q)
    img, err = generate_image_from_prompt(ad_prompt)

    # Serialize updated profile back to cookie string
    new_cookie_val = _dumps_cookie(profile)

    if err:
        return (
            new_cookie_val,
            f"Error generating image for '{q}': {err}",
            None,
            gallery,
            ad_prompt,
            search_counter,
        )

    gallery = gallery or []
    gallery.insert(0, img)

    status = f"Ad generated for '{q}'."
    return new_cookie_val, status, img, gallery, ad_prompt, (search_counter + 1)


def handle_rotate(
    cookie_shadow: str,
    gallery: List[Image.Image],
    search_counter: int,
    rotation_seen: int,
    request: gr.Request,
):
    """
    Runs every 20s. Only rotates if user hasn't searched since last rotation.
    Keeps the current ad visible when skipping/errors.
    """

    # If a new search happened since last rotation, skip but KEEP current ad
    if search_counter != rotation_seen:
        return (
            cookie_shadow,
            gallery[0][0],
            gallery,
            f"Waiting for user activity to settle...",
            "…",
            search_counter,  # sync the seen counter forward
        )

    # Load best-known profile
    req_cookie_val = request.cookies.get(COOKIE_NAME)
    profile = _merge_profiles(
        _loads_cookie(req_cookie_val), _loads_cookie(cookie_shadow)
    )

    # Need at least one past search — keep current ad
    if not profile.get("search_history"):
        return (
            cookie_shadow,
            None,
            gallery,
            "No searches yet; rotation paused.",
            "…",
            rotation_seen,
        )

    focus_query = profile["search_history"][0]
    cont_prompt = build_continuation_prompt(profile, focus_query)

    # Feed current ad to keep visual continuity
    img, err = generate_continuation_image(cont_prompt, gallery[0][0])

    # On error, keep current ad
    if err or img is None:
        return (
            cookie_shadow,
            gallery[0][0],
            gallery,
            f"Rotation paused {err}, {type(gallery[0][0])}.",
            cont_prompt,
            rotation_seen,
        )

    # Success: prepend new ad and show it
    gallery = gallery or []
    gallery.insert(0, img)

    status = f"Auto-rotated continuation for '{focus_query}'."
    return cookie_shadow, img, gallery, status, cont_prompt, rotation_seen


# ---------- JS: write/read cookie ----------
SET_COOKIE_JS = """
(cookie_name, cookie_value) => {
  if (!cookie_value) return "ok";
  const exp = new Date(Date.now() + 2*60*60*1000).toUTCString(); // 2h
  document.cookie = `${cookie_name}=${cookie_value}; path=/; SameSite=None; Secure; expires=${exp}`;
  return "ok";
}
"""

READ_COOKIE_JS = """
(cookie_name) => {
  const name = cookie_name + "=";
  const parts = document.cookie.split(';');
  for (let c of parts) {
    c = c.trim();
    if (c.startsWith(name)) return c.substring(name.length);
  }
  return "";
}
"""


# ---------- UI (clean) ----------
with gr.Blocks(
    theme=gr.themes.Soft(), fill_height=True, analytics_enabled=False
) as demo:
    gr.Markdown("# 🛒 Demo Shop — Personalized Ads")

    with gr.Row():
        search_bar = gr.Textbox(
            label="Search for products...",
            placeholder="e.g., football, coffee mug, running shoes",
            lines=1,
        )
        search_btn = gr.Button("Search")

    with gr.Row():
        status = gr.Markdown("")

    with gr.Row():
        latest_ad = gr.Image(label="Latest Ad", height=420)
        ad_prompt = gr.Textbox(
            label="Ad Prompt (sent to model)", lines=4, interactive=False
        )

    gallery = gr.Gallery(label="Recent Ads", columns=4, height=300)

    # Hidden plumbing
    cookie_name_state = gr.State(COOKIE_NAME)
    cookie_shadow = gr.Textbox(visible=False)
    sync_status = gr.Textbox(visible=False)
    gallery_state = gr.State([])
    search_counter = gr.State(0)  # increments on each user search
    rotation_seen = gr.State(0)  # last counter value the rotator saw

    # Preload shadow cookie on page load
    demo.load(
        fn=None, js=READ_COOKIE_JS, inputs=[cookie_name_state], outputs=[cookie_shadow]
    )

    # User search
    search_btn.click(
        fn=handle_search,
        inputs=[search_bar, cookie_shadow, gallery_state, search_counter],
        outputs=[cookie_shadow, status, latest_ad, gallery, ad_prompt, search_counter],
        api_name="search",
    )

    # Keep gallery_state in sync after manual searches
    def _sync_gallery(g):
        return g

    search_btn.click(_sync_gallery, inputs=[gallery], outputs=[gallery_state])

    # Write cookie to browser whenever it changes
    cookie_shadow.change(
        fn=None,
        js=SET_COOKIE_JS,
        inputs=[cookie_name_state, cookie_shadow],
        outputs=sync_status,
    )

    # Periodic rotation using a Timer (every 15 seconds)
    rotator = gr.Timer(15.0)
    rotator.tick(
        fn=handle_rotate,
        inputs=[cookie_shadow, gallery_state, search_counter, rotation_seen],
        outputs=[cookie_shadow, latest_ad, gallery, status, ad_prompt, rotation_seen],
    )

    # Keep gallery_state aligned after each rotation tick
    def _sync_after_rotate(g):
        return g

    rotator.tick(_sync_after_rotate, inputs=[gallery], outputs=[gallery_state])

if __name__ == "__main__":
    demo.launch(share=True)
