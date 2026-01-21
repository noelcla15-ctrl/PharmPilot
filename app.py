import streamlit as st
import psycopg2
import google.generativeai as genai
from datetime import date, timedelta
import fitz  # PyMuPDF
import PIL.Image
import time
import pandas as pd
import os
import shutil
import stat
import json
import requests
import base64
from io import BytesIO
from openai import OpenAI
from contextlib import contextmanager

# ==========================================
# ✅ ENV DETECTION + PROVIDER ORDER
# ==========================================
def running_on_streamlit_cloud() -> bool:
    return os.environ.get("STREAMLIT_CLOUD", "").lower() == "true" or bool(os.environ.get("STREAMLIT_SERVER_HEADLESS"))

IS_CLOUD = running_on_streamlit_cloud()

# ==========================================
# ⚙️ CONFIGURATION & THEME ENGINE
# ==========================================
st.set_page_config(page_title="PharmPilot", page_icon="💊", layout="wide")

LIGHT_THEME = """
<style>
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div, label { color: #000000 !important; }
    .stApp { background-color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #F0F2F6; border-right: 1px solid #E6E6E6; }
    input, textarea, select, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important; color: #000000 !important;
    }
    .flashcard { background-color: white; border: 1px solid #E0E0E0; border-left: 6px solid #4F8BF9; color: #000000 !important; }
    .flashcard-back { background-color: #eef6ff; border-left: 6px solid #00c853; color: #000000 !important; }
    div[data-testid="stExpander"] { background-color: #FFFFFF !important; border: 1px solid #E0E0E0; }
</style>
"""

DARK_THEME = """
<style>
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div, label { color: #E0E0E0 !important; }
    .stApp { background-color: #0E1117; }
    [data-testid="stSidebar"] { background-color: #262730; border-right: 1px solid #333; }
    input, textarea, select { background-color: #1E1E1E !important; color: #E0E0E0 !important; }
    .stSelectbox div[data-baseweb="select"] > div { background-color: #1E1E1E !important; color: #E0E0E0 !important; }
    .flashcard {
        background-color: #1E1E1E;
        padding: 40px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid #333; border-left: 6px solid #4F8BF9;
        font-size: 20px; margin-bottom: 20px; color: #E0E0E0 !important;
    }
    .flashcard-back {
        background-color: #162B1E;
        border-left: 6px solid #00c853;
        color: #E0E0E0 !important;
    }
    div[data-testid="stExpander"] { background-color: #1E1E1E !important; border: 1px solid #333; }
    button[data-baseweb="tab"] { background-color: transparent !important; }
</style>
"""

SHARED_CSS = """
<style>
    button p, button div { color: inherit !important; }
    .stButton>button { border-radius: 8px; height: 3em; font-weight: 600; border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.2); transition: transform 0.1s; }
    .stButton>button:active { transform: scale(0.98); }
</style>
"""

st.sidebar.title("💊 PharmPilot")
dark_mode = st.sidebar.toggle("🌗 Dark Mode", value=True)
st.markdown((DARK_THEME if dark_mode else LIGHT_THEME) + SHARED_CSS, unsafe_allow_html=True)
st.sidebar.markdown("---")

# ==========================================
# ⚙️ SECRETS & SETUP
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    DB_URL = st.secrets["SUPABASE_DB_URL"]
    
    # ... (Supabase secrets remain the same) ...
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
    SUPABASE_BUCKET = st.secrets.get("SUPABASE_STORAGE_BUCKET", "pharmpilot")
    SUPABASE_PDF_PREFIX = st.secrets.get("SUPABASE_PDF_PREFIX", "lectures")
    
    # OpenAI fallback
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
    OPENAI_TEXT_MODEL = st.secrets.get("OPENAI_TEXT_MODEL", "gpt-4o-mini")
    OPENAI_VISION_MODEL = st.secrets.get("OPENAI_VISION_MODEL", "gpt-4o-mini")
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    # Local Ollama
    OLLAMA_ENABLED = bool(st.secrets.get("OLLAMA_ENABLED", True))  # Default True for you
    OLLAMA_URL = st.secrets.get("OLLAMA_URL", "http://127.0.0.1:11434")
    OLLAMA_TEXT_MODEL = st.secrets.get("OLLAMA_TEXT_MODEL", "qwen2.5:7b-instruct")
    OLLAMA_VISION_MODEL = st.secrets.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")

except Exception:
    st.error("Missing Secrets! Check .streamlit/secrets.toml")
    st.stop()

# ✅ UPDATED PRIORITY: Local First -> Then Cloud Fallback
PROVIDER_ORDER = []
if OLLAMA_ENABLED:
    PROVIDER_ORDER.append("ollama")
PROVIDER_ORDER.extend(["gemini", "openai"])

# ✅ GEMINI 2.5 CONFIGURATION
genai.configure(api_key=GOOGLE_API_KEY)

# 2.5 models are strict; we must explicitly disable filters for medical study content
safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}

# Using your preferred 2.5 model (Stable release)
# Note: If 2.5 continues to fail, try "gemini-3-flash" (Dec 2025 release)
flash_model = genai.GenerativeModel("gemini-2.5-flash", safety_settings=safety_settings)
quiz_model = genai.GenerativeModel("gemini-2.5-pro", safety_settings=safety_settings)

SLIDE_DIR = "lecture_slides"
os.makedirs(SLIDE_DIR, exist_ok=True)
# ==========================================
# 🧠 ROBUST JSON PARSER (Fixes AI "Chatter")
# ==========================================
import re

def parse_json_response(response_text, payload_name):
    if not response_text:
        return []
    
    # 1. Strip markdown code blocks
    text = response_text.replace("```json", "").replace("```", "").strip()
    
    # 2. Naive attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Regex Rescue: Find the largest JSON list [...] or object {...}
    # This ignores "Here is your JSON:" intros that AI loves to add
    try:
        match_list = re.search(r'\[.*\]', text, re.DOTALL)
        if match_list:
            return json.loads(match_list.group())
            
        match_obj = re.search(r'\{.*\}', text, re.DOTALL)
        if match_obj:
            return json.loads(match_obj.group())
    except Exception:
        pass
        
    # If we get here, the AI output was likely malformed or empty
    # Consider logging 'text' here for debugging
    return []

# ==========================================
# ✅ DB: cached connection + safe cursor context
#    Fixes "idle in transaction" by ensuring cursors close and commits happen.
# ==========================================
@st.cache_resource
def get_db_connection():
    return psycopg2.connect(DB_URL)

def _healthcheck(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT 1;")

def get_conn():
    conn = get_db_connection()
    try:
        if conn is None or conn.closed != 0:
            raise psycopg2.InterfaceError("Cached connection was closed")
        _healthcheck(conn)
        return conn
    except (psycopg2.InterfaceError, psycopg2.OperationalError):
        try:
            get_db_connection.clear()
        except Exception:
            pass
        conn = get_db_connection()
        _healthcheck(conn)
        return conn

@contextmanager
def db_cursor(commit: bool = False):
    conn = get_conn()
    cur = conn.cursor()
    try:
        yield conn, cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass

# ==========================================
# 🧹 SCHEMA ENSURE (pdf_path + SR improvements columns)
# ==========================================
def ensure_schema():
    with db_cursor(commit=True) as (_conn, c):
        # lectures.pdf_path
        c.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='lectures' AND column_name='pdf_path'
            );
            """
        )
        if not c.fetchone()[0]:
            c.execute("ALTER TABLE lectures ADD COLUMN pdf_path TEXT;")

        # cards.lapses + cards.last_reviewed (minimal SR upgrade without breaking existing rows)
        c.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='cards' AND column_name='lapses'
            );
            """
        )
        if not c.fetchone()[0]:
            c.execute("ALTER TABLE cards ADD COLUMN lapses INT DEFAULT 0;")

        c.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='cards' AND column_name='last_reviewed'
            );
            """
        )
        if not c.fetchone()[0]:
            c.execute("ALTER TABLE cards ADD COLUMN last_reviewed DATE;")

ensure_schema()

# ==========================================
# 🧹 AUTOMATIC DATABASE CLEANUP (runs once per day per session)
# ==========================================
def run_cleanup_once_per_day():
    today_iso = date.today().isoformat()
    if st.session_state.get("last_cleanup") == today_iso:
        return
    try:
        with db_cursor(commit=True) as (_conn, c):
            # Optional: if you created this function in DB
            c.execute("select public.cleanup_old_cards();")
        st.session_state["last_cleanup"] = today_iso
    except Exception as e:
        # keep silent in UI; avoid breaking app
        print("Cleanup failed:", e)

run_cleanup_once_per_day()

# ==========================================
# ✅ Supabase Storage (PDF-only) via REST
# ==========================================
def _supabase_storage_headers(content_type: str | None = None):
    h = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    }
    if content_type:
        h["Content-Type"] = content_type
    return h

def supabase_pdf_object_path(lecture_id: int) -> str:
    return f"{SUPABASE_PDF_PREFIX}/{lecture_id}.pdf"

def upload_pdf_to_supabase(lecture_id: int, pdf_bytes: bytes) -> str:
    obj_path = supabase_pdf_object_path(lecture_id)
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{obj_path}"
    r = requests.post(
        url,
        headers=_supabase_storage_headers("application/pdf"),
        params={"upsert": "true"},
        data=pdf_bytes,
        timeout=60,
    )
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"Supabase upload failed ({r.status_code}): {r.text}")
    return obj_path

def download_pdf_from_supabase(object_path: str) -> bytes:
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{object_path}"
    r = requests.get(url, headers=_supabase_storage_headers(), timeout=60)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"Supabase download failed ({r.status_code}): {r.text}")
    return r.content

def delete_pdf_from_supabase(object_path: str) -> bool:
    """
    Deletes an object from Supabase storage.
    NOTE: This will only work if your bucket policies allow deletes with anon key.
    """
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{object_path}"
    r = requests.delete(url, headers=_supabase_storage_headers(), timeout=30)
    return 200 <= r.status_code < 300

# ==========================================
# 🧠 JSON PARSER (shared)
# ==========================================
def parse_json_response(response_text, payload_name):
    clean_text = (response_text or "").replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        start = clean_text.find("[")
        end = clean_text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(clean_text[start:end + 1])
        start = clean_text.find("{")
        end = clean_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(clean_text[start:end + 1])
        raise ValueError(f"Unable to parse {payload_name} JSON response.")

# ==========================================
# 🧠 OLLAMA HELPERS (LOCAL AI)
# ==========================================
def ollama_is_up(ollama_url: str) -> bool:
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def pil_to_base64_png(pil_img) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def ollama_generate_text(prompt: str, model: str, temperature=0.2, timeout=180) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "")

def ollama_generate_vision(prompt: str, pil_images: list, model: str, temperature=0.2, timeout=240) -> str:
    images_b64 = [pil_to_base64_png(img) for img in pil_images]
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "")

# ==========================================
# 🧠 OPENAI HELPERS (CLOUD FALLBACK)
# ==========================================
def openai_generate_text(prompt: str, model: str, timeout=180) -> str:
    if not openai_client:
        raise RuntimeError("OpenAI client not configured (missing OPENAI_API_KEY).")
    resp = openai_client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        timeout=timeout,
    )
    out_text = ""
    for item in (resp.output or []):
        for part in (getattr(item, "content", []) or []):
            if getattr(part, "type", "") in ("output_text", "text"):
                out_text += getattr(part, "text", "") or ""
    return out_text.strip()

def openai_generate_vision(prompt: str, pil_images: list, model: str, timeout=240) -> str:
    if not openai_client:
        raise RuntimeError("OpenAI client not configured (missing OPENAI_API_KEY).")

    def pil_to_data_url(img):
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    content = [{"type": "input_text", "text": prompt}]
    for img in pil_images:
        content.append({"type": "input_image", "image_url": pil_to_data_url(img)})

    resp = openai_client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        timeout=timeout,
    )

    out_text = ""
    for item in (resp.output or []):
        for part in (getattr(item, "content", []) or []):
            if getattr(part, "type", "") in ("output_text", "text"):
                out_text += getattr(part, "text", "") or ""
    return out_text.strip()

# ==========================================
# 🧠 PDF UTILITIES (text + images)
# ==========================================
def extract_slide_texts_from_pdf_bytes(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page in doc:
        txt = page.get_text("text") or ""
        txt = "\n".join([line.strip() for line in txt.splitlines() if line.strip()])
        texts.append(txt)
    return texts

def save_slides_locally_from_pdf_bytes(pdf_bytes: bytes, lecture_id: int, dpi=250):
    slide_images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    lec_path = os.path.join(SLIDE_DIR, str(lecture_id))

    def on_rm_error(_func, path, _exc_info):
        os.chmod(path, stat.S_IWRITE)
        os.unlink(path)

    if os.path.exists(lec_path):
        try:
            shutil.rmtree(lec_path, onerror=on_rm_error)
        except Exception:
            pass

    os.makedirs(lec_path, exist_ok=True)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img_path = os.path.join(lec_path, f"slide_{i}.png")
        pix.save(img_path)
        slide_images.append(PIL.Image.open(img_path))
    return slide_images

def slide_needs_vision(slide_text: str, min_chars=140):
    return len((slide_text or "").strip()) < min_chars

def load_slide_cache(lecture_id: int):
    cache_path = os.path.join(SLIDE_DIR, str(lecture_id), "slide_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_slide_cache(lecture_id: int, cache_obj):
    cache_path = os.path.join(SLIDE_DIR, str(lecture_id), "slide_cache.json")
    # FIX: Ensure directory exists just in case
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_obj, f, ensure_ascii=False, indent=2)

def build_slide_cache(lecture_id: int, slide_texts):
    cache = []
    for i, t in enumerate(slide_texts):
        cache.append({"i": i, "text": t, "needs_vision": slide_needs_vision(t), "image_notes": ""})
    save_slide_cache(lecture_id, cache)
    return cache

def ensure_image_notes_for_batch(lecture_id, batch_indices, batch_images, cache):
    if not (OLLAMA_ENABLED and ollama_is_up(OLLAMA_URL)):
        return cache

    to_process = []
    to_process_slide_idxs = []

    for pos, slide_i in enumerate(batch_indices):
        entry = cache[slide_i]
        if entry.get("needs_vision") and not entry.get("image_notes"):
            to_process.append(batch_images[pos])
            to_process_slide_idxs.append(slide_i)

    if not to_process:
        return cache

    GROUP = 2
    for start in range(0, len(to_process), GROUP):
        imgs = to_process[start:start + GROUP]
        slide_idxs = to_process_slide_idxs[start:start + GROUP]

        prompt = """
You are interpreting pharmacy lecture slides with diagrams/figures.
For each slide image, write concise notes:
- what the figure/diagram is showing
- key labels/axes interpretation
- common exam traps / what a professor might ask
Return RAW JSON only:
[{"notes":"..."}, {"notes":"..."}]
Return items in the same order as the images.
""".strip()

        try:
            resp = ollama_generate_vision(prompt, imgs, model=OLLAMA_VISION_MODEL, temperature=0.1, timeout=240)
            notes = parse_json_response(resp, "image_notes")
            if isinstance(notes, list) and notes:
                for k, item in enumerate(notes):
                    n = (item.get("notes", "") or "").strip() if isinstance(item, dict) else ""
                    if k < len(slide_idxs):
                        cache[slide_idxs[k]]["image_notes"] = n
        except Exception:
            pass

    save_slide_cache(lecture_id, cache)
    return cache

# ==========================================
# ✅ Ensure local cache exists by downloading PDF from Supabase when needed
# ==========================================
def objectives_path(lecture_id: int):
    return os.path.join(SLIDE_DIR, str(lecture_id), "objectives.json")

def save_objectives(lecture_id: int, objectives_list):
    os.makedirs(os.path.join(SLIDE_DIR, str(lecture_id)), exist_ok=True)
    with open(objectives_path(lecture_id), "w", encoding="utf-8") as f:
        json.dump({"objectives": objectives_list}, f, ensure_ascii=False, indent=2)

def load_objectives(lecture_id: int):
    p = objectives_path(lecture_id)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f).get("objectives", [])
        except Exception:
            return []
    return []

def extract_objectives_from_slide_texts(slide_texts, max_slides=8):
    early = "\n\n".join([f"Slide {i+1}:\n{slide_texts[i]}" for i in range(min(max_slides, len(slide_texts)))])
    prompt = f"""
Extract the lecture LEARNING OBJECTIVES from the early slides below.

Rules:
- If objectives are explicitly listed, extract them verbatim-ish (cleaned).
- If not explicitly listed, infer 6–10 likely objectives based on repeated themes.
- Bias toward: definitions, formulas, decision rules, models, drug facts, comparisons.

Return RAW JSON ONLY:
{{"objectives":["...","..."]}}

SLIDES:
{early}
""".strip()

    try:
        resp = flash_model.generate_content(prompt)
        data = parse_json_response(resp.text, "objectives")
        objs = data.get("objectives", [])
        return [o.strip() for o in objs if isinstance(o, str) and o.strip()]
    except Exception:
        if openai_client:
            try:
                resp_text = openai_generate_text(prompt, model=OPENAI_TEXT_MODEL, timeout=120)
                data = parse_json_response(resp_text, "objectives")
                objs = data.get("objectives", [])
                return [o.strip() for o in objs if isinstance(o, str) and o.strip()]
            except Exception:
                pass
    return []

def objectives_to_string(obj_input, lecture_id: int):
    if obj_input and obj_input.strip():
        return obj_input.strip()
    stored = load_objectives(lecture_id)
    if stored:
        return "\n".join(f"- {x}" for x in stored)
    return "No objectives provided. Prioritize definitions, formulas, models, comparisons, and drugs."

def ensure_local_assets_for_lecture(lecture_id: int):
    lec_path = os.path.join(SLIDE_DIR, str(lecture_id))
    cache = load_slide_cache(lecture_id)

    have_pngs = os.path.exists(lec_path) and any(
        f.endswith(".png") and f.startswith("slide_") for f in os.listdir(lec_path)
    )

    if cache is not None and have_pngs:
        return cache  # FIX: Return the existing cache

    with db_cursor() as (_conn, c):
        c.execute("SELECT pdf_path FROM lectures WHERE id=%s", (lecture_id,))
        row = c.fetchone()
    pdf_path = row[0] if row else None
    if not pdf_path:
        raise RuntimeError("Lecture PDF not found (missing lectures.pdf_path). Re-upload that lecture.")

    pdf_bytes = download_pdf_from_supabase(pdf_path)
    _ = save_slides_locally_from_pdf_bytes(pdf_bytes, lecture_id, dpi=250)
    slide_texts = extract_slide_texts_from_pdf_bytes(pdf_bytes)
    
    # FIX: Build cache and assign it
    cache = build_slide_cache(lecture_id, slide_texts)

    if not load_objectives(lecture_id):
        auto_objs = extract_objectives_from_slide_texts(slide_texts, max_slides=8)
        save_objectives(lecture_id, auto_objs)

    # FIX: Return the newly built cache
    return cache

# ==========================================
# ✅ HIGH-YIELD GATING
# ==========================================
def build_slides_blob(batch_entries, start_idx):
    parts = []
    for k, entry in enumerate(batch_entries):
        slide_no = start_idx + k + 1
        txt = entry.get("text", "") or ""
        img_notes = entry.get("image_notes", "") or ""
        parts.append(f"--- Slide {slide_no} ---\nTEXT:\n{txt}\n\nIMAGE_NOTES:\n{img_notes}".strip())
    return "\n\n".join(parts)

def extract_high_yield(batch_entries, objectives_str, start_idx, provider="gemini"):
    slides_blob = build_slides_blob(batch_entries, start_idx)

    prompt = f"""
You are a strict pharmacy exam content curator.

OBJECTIVES (use these to decide relevance):
{objectives_str}

HARD EXCLUSIONS (mark NOT relevant if mostly these):
- instructor contact info, emails, phone numbers, office hours
- class logistics, dates, assignments, citations/references pages
- admin content, generic background, filler
- isolated trivia numbers unless explicitly objective-relevant

RELEVANT if aligned with objectives and is testable:
- definitions, formulas, decision rules
- key comparisons
- drugs (MOA/uses/AE/contraindications/pearls)
- model mechanics (states, transitions, costs, probabilities)

Return RAW JSON ONLY:
[
  {
    "slide": <int>,
    "relevant": true/false,
    "why": "...",
    "key_terms": ["..."],
    "key_points": ["..."]
  }
]

SLIDES:
{slides_blob}
""".strip()

    if provider == "ollama":
        resp = ollama_generate_text(prompt, model=OLLAMA_TEXT_MODEL, temperature=0.1, timeout=180)
        data = parse_json_response(resp, "high_yield")
        return data if isinstance(data, list) else []
    if provider == "openai":
        resp = openai_generate_text(prompt, model=OPENAI_TEXT_MODEL, timeout=180)
        data = parse_json_response(resp, "high_yield")
        return data if isinstance(data, list) else []

    resp = flash_model.generate_content(prompt)
    data = parse_json_response(resp.text, "high_yield")
    return data if isinstance(data, list) else []

def condensed_notes_from_high_yield(high_yield_items):
    condensed = []
    for item in high_yield_items:
        if not isinstance(item, dict):
            continue
        if not item.get("relevant"):
            continue
        kp = item.get("key_points") or []
        kt = item.get("key_terms") or []
        slide_no = item.get("slide")
        if kp:
            condensed.append(f"Slide {slide_no}:\nKEY_TERMS: {kt}\nKEY_POINTS:\n- " + "\n- ".join(kp))
    return "\n\n".join(condensed).strip()

def generate_cards_from_notes(notes_blob, objectives_str, provider="gemini"):
    if not notes_blob:
        return []

    prompt = f"""
You are a Pharmacy Professor writing exam-grade flashcards.

OBJECTIVES:
{objectives_str}

ONLY use the curated HIGH-YIELD NOTES below. Do NOT invent facts.
Make cards primarily in these styles:
- Definitions / distinctions
- Formulas / interpretation
- Model mechanics
- Drug facts (MOA, indications, major adverse effects, pearls)

Avoid:
- instructor contact info, emails, logistics
- trivial stats unless clearly objective-relevant

Return RAW JSON ONLY:
[{{"front":"...","back":"..."}}]

HIGH-YIELD NOTES:
{notes_blob}
""".strip()

    if provider == "ollama":
        resp = ollama_generate_text(prompt, model=OLLAMA_TEXT_MODEL, temperature=0.2, timeout=180)
        data = parse_json_response(resp, "flashcards")
        if isinstance(data, list):
            return [(x["front"], x["back"]) for x in data if isinstance(x, dict) and "front" in x and "back" in x]
        return []

    if provider == "openai":
        resp = openai_generate_text(prompt, model=OPENAI_TEXT_MODEL, timeout=180)
        data = parse_json_response(resp, "flashcards")
        if isinstance(data, list):
            return [(x["front"], x["back"]) for x in data if isinstance(x, dict) and "front" in x and "back" in x]
        return []

    resp = flash_model.generate_content(prompt)
    data = parse_json_response(resp.text, "flashcards")
    if isinstance(data, list):
        return [(x["front"], x["back"]) for x in data if isinstance(x, dict) and "front" in x and "back" in x]
    return []

# ==========================================
# ☁️ GEMINI QUIZ
# ==========================================
def generate_interactive_quiz_gemini(images):
    prompt = """
Create a 5-question multiple choice quiz based on these slides.
Target Audience: Pharmacy Students (NAPLEX level).
IMPORTANT: Return ONLY a JSON list. Do not use Markdown blocks.
Structure: [ { "question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct_index": 0, "explanation": "..." } ]
"""
    try:
        content = [prompt] + images
        response = quiz_model.generate_content(content)
        return parse_json_response(response.text, "quiz")
    except Exception as e:
        return [{"error": str(e)}]

def generate_quiz_local_textfirst(batch_entries):
    slides_blob = build_slides_blob(batch_entries, start_idx=0)
    prompt = f"""
Create a 5-question multiple choice quiz based on these slides.
Target audience: Pharmacy students (NAPLEX level).

Return RAW JSON ONLY:
[
  {
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correct_index": 0,
    "explanation": "..."
  }
]

SLIDES:
{slides_blob}
""".strip()
    resp = ollama_generate_text(prompt, model=OLLAMA_TEXT_MODEL, temperature=0.2, timeout=180)
    return parse_json_response(resp, "quiz")

# ==========================================
# 🧠 HYBRID WRAPPERS
# ==========================================
def generate_cards_hybrid(lecture_id, batch_indices, batch_images, cache, objectives_input, start_idx):
    cache = ensure_image_notes_for_batch(lecture_id, batch_indices, batch_images, cache)
    batch_entries = [cache[i] for i in batch_indices]
    objectives_str = objectives_to_string(objectives_input, lecture_id)

    last_err = None
    for provider in PROVIDER_ORDER:
        try:
            if provider == "ollama":
                if OLLAMA_ENABLED and ollama_is_up(OLLAMA_URL):
                    hy = extract_high_yield(batch_entries, objectives_str, start_idx, provider="ollama")
                    notes = condensed_notes_from_high_yield(hy)
                    cards = generate_cards_from_notes(notes, objectives_str, provider="ollama")
                    if cards:
                        return cards, None
                    last_err = "Ollama produced no cards."

            elif provider == "gemini":
                hy = extract_high_yield(batch_entries, objectives_str, start_idx, provider="gemini")
                notes = condensed_notes_from_high_yield(hy)
                cards = generate_cards_from_notes(notes, objectives_str, provider="gemini")
                if cards:
                    return cards, None
                last_err = "Gemini produced no cards."

            elif provider == "openai":
                if openai_client:
                    hy = extract_high_yield(batch_entries, objectives_str, start_idx, provider="openai")
                    notes = condensed_notes_from_high_yield(hy)
                    cards = generate_cards_from_notes(notes, objectives_str, provider="openai")
                    if cards:
                        return cards, None
                    last_err = "OpenAI produced no cards."
                else:
                    last_err = "OpenAI not configured."

        except Exception as e:
            last_err = f"{provider} failed: {e}"

    return [], f"All providers failed or returned no cards. Last: {last_err}"

def generate_quiz_hybrid(lecture_id, batch_indices, batch_images, cache):
    cache = ensure_image_notes_for_batch(lecture_id, batch_indices, batch_images, cache)
    batch_entries = [cache[i] for i in batch_indices]

    last_err = None
    for provider in PROVIDER_ORDER:
        try:
            if provider == "ollama":
                if OLLAMA_ENABLED and ollama_is_up(OLLAMA_URL):
                    return generate_quiz_local_textfirst(batch_entries)
                last_err = "Ollama disabled/unreachable."

            elif provider == "gemini":
                q = generate_interactive_quiz_gemini(batch_images)
                if isinstance(q, list) and q and isinstance(q[0], dict) and "error" not in q[0]:
                    return q
                last_err = f"Gemini quiz error: {q[0].get('error') if isinstance(q, list) and q else 'unknown'}"

            elif provider == "openai":
                if openai_client:
                    prompt = """
Create a 5-question multiple choice quiz based on these slides.
Target Audience: Pharmacy Students (NAPLEX level).
IMPORTANT: Return ONLY a JSON list. Do not use Markdown blocks.
Structure: [ { "question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct_index": 0, "explanation": "..." } ]
""".strip()
                    resp_text = openai_generate_vision(prompt, batch_images, model=OPENAI_VISION_MODEL)
                    return parse_json_response(resp_text, "quiz")
                last_err = "OpenAI not configured."

        except Exception as e:
            last_err = f"{provider} failed: {e}"

    return [{"error": f"All providers failed. Last: {last_err}"}]

# ==========================================
# ✅ SPACED REPETITION OVERHAUL (exam-aware SM-2-ish)
# ==========================================
def sr_compute_next(interval: int | None, ease: float | None, review_count: int | None, lapses: int | None, quality: int):
    """
    Returns (new_interval_days:int, new_ease:float, new_lapses:int)

    quality: 0=Again, 3=Hard, 4=Good, 5=Easy
    - Fixes "Hard" behavior: Hard should not push interval out more than Good.
    - Adds lapse tracking.
    - Handles new cards with a short learning ramp (day-granularity).
    """
    interval = int(interval) if interval and interval > 0 else 1
    ease = float(ease) if ease else 2.3
    review_count = int(review_count) if review_count is not None else 0
    lapses = int(lapses) if lapses is not None else 0

    # clamp ease
    ease = max(1.3, min(3.0, ease))

    # NEW CARD BOOTSTRAP (first successful reviews)
    # Day-granularity learning ramp:
    # - Good => 3 days, Easy => 4 days, Hard => 1 day, Again => 1 day
    if review_count == 0:
        if quality == 0:
            return 1, max(1.3, ease - 0.2), lapses + 1
        if quality == 3:
            return 1, max(1.3, ease - 0.1), lapses
        if quality == 4:
            return 3, ease, lapses
        if quality == 5:
            return 4, min(3.0, ease + 0.1), lapses

    # REVIEW PHASE
    if quality == 0:
        lapses += 1
        ease = max(1.3, ease - 0.2)
        # relearn: back tomorrow
        return 1, ease, lapses

    if quality == 3:
        # Hard should be smaller than Good; also slightly penalize ease
        ease = max(1.3, ease - 0.15)
        good_int = max(2, int(round(interval * ease)))
        hard_int = max(1, int(round(interval * 1.2)))
        # ensure hard < good when possible
        if hard_int >= good_int:
            hard_int = max(1, good_int - 1)
        return hard_int, ease, lapses

    if quality == 4:
        new_int = max(2, int(round(interval * ease)))
        return new_int, ease, lapses

    if quality == 5:
        ease = min(3.0, ease + 0.15)
        new_int = max(3, int(round(interval * ease * 1.3)))
        return new_int, ease, lapses

    # fallback
    return max(1, interval), ease, lapses

def apply_exam_cap(new_interval: int, exam_date: date | None, today: date):
    if exam_date and (exam_date - today).days > 0:
        days_limit = (exam_date - today).days
        if new_interval >= days_limit:
            return max(1, days_limit - 1)
    return new_interval

# ==========================================
# 🖥️ UI STATE
# ==========================================
if "main_nav" not in st.session_state:
    st.session_state.main_nav = "Review"
if "show" not in st.session_state:
    st.session_state.show = False
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "read_idx" not in st.session_state:
    st.session_state.read_idx = 0

nav = st.sidebar.radio("Menu", ["Review", "Library", "Active Learning", "Editor"], key="main_nav")

def open_lecture_callback(lid):
    st.session_state.active_lecture_id = lid
    st.session_state.main_nav = "Active Learning"

# ------------------------------------------
# 1. REVIEW DASHBOARD
# ------------------------------------------
if nav == "Review":
    st.title("🧠 Study Center")
    today = date.today()

    if "session_active" not in st.session_state:
        st.session_state.session_active = False
    if "streak" not in st.session_state:
        st.session_state.streak = 0
    if "last_completion_date" not in st.session_state:
        st.session_state.last_completion_date = None
    if "missed_content" not in st.session_state:
        st.session_state.missed_content = []

    with db_cursor() as (conn, c):
        c.execute("""
            SELECT c.id, c.front, c.back, c.interval, c.ease, c.review_count, c.lapses,
                   l.id, l.name, e.exam_date, e.name, e.id
            FROM cards c
            JOIN lectures l ON c.lecture_id = l.id
            JOIN exams e ON l.exam_id = e.id
            WHERE c.next_review <= %s
            ORDER BY c.next_review ASC LIMIT 50
        """, (today,))
        cards_due = c.fetchall()

        c.execute("SELECT name, exam_date FROM exams WHERE exam_date >= %s ORDER BY exam_date ASC LIMIT 1", (today,))
        next_ex = c.fetchone()

    if not st.session_state.session_active:
        if not cards_due:
            st.balloons()
            st.success("🎉 All caught up for today!")
            st.metric("🔥 Current Streak", f"{st.session_state.streak} Days")
        else:
            total_due = len(cards_due)

            st.markdown("### 📊 Session Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cards Due", total_due)
            with col2:
                if next_ex:
                    days_left = (next_ex[1] - today).days
                    st.metric("Next Exam", f"{days_left} Days", next_ex[0])
                else:
                    st.metric("Next Exam", "None Set")
            with col3:
                avg_speed = 10 if "global_avg_speed" not in st.session_state else st.session_state.global_avg_speed
                st.metric("Est. Time", f"~{int(total_due * avg_speed // 60)} min")
            with col4:
                st.metric("🔥 Streak", f"{st.session_state.streak} Days")

            st.markdown("---")
            if st.button("🚀 Start Review Session", type="primary", use_container_width=True):
                st.session_state.session_active = True
                st.session_state.idx = 0
                st.session_state.total_seconds = 0
                st.session_state.trouble_lectures = {}
                st.session_state.missed_content = []
                st.session_state.session_start_time = time.time()
                st.rerun()

    else:
        total_cards = len(cards_due)
        if st.session_state.idx < total_cards:
            remaining = total_cards - st.session_state.idx
            if st.session_state.idx > 0:
                avg_speed = st.session_state.total_seconds / st.session_state.idx
                time_display = f"⏱️ {int((avg_speed * remaining)//60)}m {int((avg_speed * remaining)%60)}s left"
            else:
                time_display = "⏱️ Calculating..."

            st.progress(st.session_state.idx / total_cards)
            c_left, c_right = st.columns(2)
            c_left.caption(f"Card {st.session_state.idx + 1} of {total_cards}")
            c_right.markdown(f"<p style='text-align:right; font-weight:bold;'>{time_display}</p>", unsafe_allow_html=True)

            (
                cid, front, back, interval, ease, revs, lapses,
                lid, lname, exam_date, exam_name, eid
            ) = cards_due[st.session_state.idx]

            if "card_load_time" not in st.session_state or not st.session_state.show:
                st.session_state.card_load_time = time.time()

            st.markdown(f'<div class="flashcard"><small>QUESTION</small><br>{front}</div>', unsafe_allow_html=True)

            if st.session_state.show:
                st.markdown(f'<div class="flashcard flashcard-back"><small>ANSWER</small><br>{back}</div>', unsafe_allow_html=True)

                def answer(quality: int):
                    # Track trouble content for Again/Hard
                    if quality in [0, 3]:
                        st.session_state.missed_content.append(f"Q: {front} | A: {back}")
                        if lid not in st.session_state.trouble_lectures:
                            st.session_state.trouble_lectures[lid] = {"count": 1, "name": lname, "exam_date": exam_date}
                        else:
                            st.session_state.trouble_lectures[lid]["count"] += 1

                    st.session_state.total_seconds += (time.time() - st.session_state.card_load_time)

                    new_interval, new_ease, new_lapses = sr_compute_next(
                        interval=interval,
                        ease=ease,
                        review_count=revs,
                        lapses=lapses,
                        quality=quality
                    )
                    new_interval = apply_exam_cap(new_interval, exam_date, today)

                    next_review = today + timedelta(days=int(new_interval))

                    with db_cursor(commit=True) as (_conn, c):
                        c.execute(
                            """
                            UPDATE cards
                            SET next_review=%s,
                                interval=%s,
                                ease=%s,
                                review_count=%s,
                                lapses=%s,
                                last_reviewed=%s
                            WHERE id=%s
                            """,
                            (next_review, int(new_interval), float(new_ease), int(revs) + 1, int(new_lapses), today, cid),
                        )

                    st.session_state.show = False
                    st.session_state.idx += 1
                    st.rerun()

                c1, c2, c3, c4 = st.columns(4)
                if c1.button("❌ Again", use_container_width=True):
                    answer(0)
                if c2.button("😓 Hard", use_container_width=True):
                    answer(3)
                if c3.button("✅ Good", use_container_width=True):
                    answer(4)
                if c4.button("🚀 Easy", use_container_width=True):
                    answer(5)
            else:
                if st.button("Show Answer", type="primary", use_container_width=True):
                    st.session_state.show = True
                    st.rerun()

        else:
            st.balloons()
            total_time = time.time() - st.session_state.session_start_time
            st.success(f"Session Finished! Total time: {int(total_time//60)}m {int(total_time%60)}s")

            if st.session_state.last_completion_date == today - timedelta(days=1):
                st.session_state.streak += 1
            elif st.session_state.last_completion_date != today:
                st.session_state.streak = 1
            st.session_state.last_completion_date = today

            if st.session_state.missed_content:
                st.markdown("### 📖 Smart Cheat Sheet")
                if st.button("🪄 Generate AI Summary of Missed Concepts"):
                    with st.spinner("Analyzing your weak spots..."):
                        missed_str = "\n".join(st.session_state.missed_content[:15])
                        prompt = (
                            "As a Pharmacy Professor, summarize these missed concepts into a one-page clinical cheat sheet. "
                            "Use bullet points and focus on high-yield exam facts:\n"
                            f"{missed_str}"
                        )
                        response = flash_model.generate_content(prompt)
                        st.markdown(
                            f'<div style="background-color:#FFF8E1; padding:20px; border-radius:10px; color:black;">{response.text}</div>',
                            unsafe_allow_html=True,
                        )

            if st.session_state.trouble_lectures:
                urgent = [v for v in st.session_state.trouble_lectures.values()
                          if v["exam_date"] and (v["exam_date"] - today).days <= 14]
                if urgent:
                    top_trouble = sorted(urgent, key=lambda x: x["count"], reverse=True)[0]
                    st.warning(f"💊 **Lecture Recommendation:** Review **{top_trouble['name']}**.")
                    if st.button("📖 Deep Dive Now", type="primary"):
                        for lid2, data in st.session_state.trouble_lectures.items():
                            if data["name"] == top_trouble["name"]:
                                open_lecture_callback(lid2)
                                st.rerun()

            if st.button("Finish & Back to Prep Room"):
                st.session_state.session_active = False
                st.rerun()

# ------------------------------------------
# 2. LIBRARY
# ------------------------------------------
elif nav == "Library":
    st.title("📂 Library")

    tab_browse, tab_upload, tab_manage = st.tabs(["📚 Browse Materials", "☁️ Upload New", "⚙️ Manage"])

    with tab_browse:
        with db_cursor() as (conn, _c):
            big_query = """
            SELECT c.name as class_name, e.name as exam_name, e.id as exam_id,
                   l.id as lecture_id, l.name as lecture_name, l.slide_count
            FROM classes c
            JOIN exams e ON e.class_id = c.id
            LEFT JOIN lectures l ON l.exam_id = e.id
            ORDER BY c.name, e.name, l.name
            """
            df = pd.read_sql(big_query, conn)

        if df.empty:
            st.info("Library is empty.")
        else:
            for class_name, class_group in df.groupby("class_name"):
                with st.expander(f"📁 {class_name}", expanded=False):
                    for exam_name, exam_group in class_group.groupby("exam_name"):
                        st.caption(f"📂 {exam_name}")
                        for _, row in exam_group.iterrows():
                            if pd.notna(row["lecture_name"]):
                                lid = int(row["lecture_id"])
                                lname = row["lecture_name"]
                                c1, c2, c3 = st.columns([0.6, 0.2, 0.2])
                                with c1:
                                    st.markdown(f"**{lname}**")
                                with c2:
                                    st.button("Open", key=f"op_{lid}", on_click=open_lecture_callback, args=(lid,))
                                with c3:
                                    if st.button("⚡ AI", key=f"rt_{lid}", help="Retry AI"):
                                        try:
                                            ensure_local_assets_for_lecture(lid)
                                        except Exception as e:
                                            st.error(str(e))
                                            st.stop()

                                        lec_path = os.path.join(SLIDE_DIR, str(lid))
                                        slides = sorted(
                                            [f for f in os.listdir(lec_path) if f.endswith(".png")],
                                            key=lambda x: int(x.split("_")[1].split(".")[0]),
                                        )
                                        images = [PIL.Image.open(os.path.join(lec_path, s)) for s in slides]

                                        cache = load_slide_cache(lid)
                                        if not cache:
                                            st.error("slide_cache.json missing and could not be rebuilt.")
                                            st.stop()

                                        stored_obj = "\n".join(f"- {x}" for x in load_objectives(lid)) or ""

                                        st.toast(f"Processing {len(images)} slides...", icon="⚡")
                                        for i in range(0, len(images), 10):
                                            batch_imgs = images[i:i + 10]
                                            batch_indices = list(range(i, min(i + 10, len(images))))
                                            new_cards, error = generate_cards_hybrid(lid, batch_indices, batch_imgs, cache, stored_obj, i)
                                            if error:
                                                st.error(f"AI flashcard note: {error}")
                                            if new_cards:
                                                with db_cursor(commit=True) as (_conn, c):
                                                    for f, b in new_cards:
                                                        c.execute(
                                                            "INSERT INTO cards (lecture_id, front, back, next_review) VALUES (%s,%s,%s,%s)",
                                                            (lid, f, b, date.today()),
                                                        )
                                        st.rerun()
                        st.divider()

    with tab_upload:
        with db_cursor() as (conn, c):
            c.execute("SELECT id, name FROM classes")
            classes = c.fetchall()

        if not classes:
            st.warning("Create a Class in 'Manage' tab first!")
        else:
            c_map = {n: i for i, n in classes}
            sel_c = st.selectbox("Select Class", list(c_map.keys()))

            with db_cursor() as (conn, c):
                c.execute("SELECT id, name FROM exams WHERE class_id=%s", (c_map[sel_c],))
                exams = c.fetchall()

            with st.expander("➕ Add New Topic"):
                new_topic = st.text_input("Topic Name")
                d_val = st.date_input("Exam Date")
                if st.button("Create Topic"):
                    if not new_topic.strip():
                        st.warning("Topic name cannot be empty.")
                    else:
                        with db_cursor(commit=True) as (_conn, c):
                            c.execute(
                                "INSERT INTO exams (class_id, name, exam_date) VALUES (%s,%s,%s)",
                                (c_map[sel_c], new_topic.strip(), d_val),
                            )
                        st.rerun()

            if exams:
                e_map = {n: i for i, n in exams}
                sel_e = st.selectbox("Select Topic", list(e_map.keys()))
                uploaded_files = st.file_uploader("Drop PDFs Here", type="pdf", accept_multiple_files=True)
                objs = st.text_area("Learning Objectives (optional — auto-extracts if blank)")

                # NEW: Checkbox to control flashcard generation
                gen_cards = st.checkbox("✨ Auto-generate Flashcards?", value=True, help="Uncheck to just upload the PDF/slides without creating cards yet.")

                if uploaded_files and st.button("🚀 Upload & Process", type="primary"):
                    status = st.status("Processing...", expanded=True)

                    for uploaded in uploaded_files:
                        status.write(f"Reading {uploaded.name}...")
                        pdf_bytes = uploaded.getvalue()

                        # Create lecture row first
                        with db_cursor(commit=True) as (_conn, c):
                            c.execute(
                                "INSERT INTO lectures (exam_id, name, slide_count, pdf_path) VALUES (%s,%s,%s,%s) RETURNING id",
                                (e_map[sel_e], uploaded.name, 0, None),
                            )
                            lid = c.fetchone()[0]

                        # Upload PDF to Supabase Storage, then store pdf_path
                        try:
                            pdf_path = upload_pdf_to_supabase(lid, pdf_bytes)
                        except Exception:
                            with db_cursor(commit=True) as (_conn, c):
                                c.execute("DELETE FROM lectures WHERE id=%s", (lid,))
                            raise

                        with db_cursor(commit=True) as (_conn, c):
                            c.execute("UPDATE lectures SET pdf_path=%s WHERE id=%s", (pdf_path, lid))

                        images = save_slides_locally_from_pdf_bytes(pdf_bytes, lid, dpi=250)
                        slide_texts = extract_slide_texts_from_pdf_bytes(pdf_bytes)
                        cache = build_slide_cache(lid, slide_texts)

                        if not (objs and objs.strip()):
                            status.write("Auto-extracting learning objectives...")
                            auto_objs = extract_objectives_from_slide_texts(slide_texts, max_slides=8)
                            save_objectives(lid, auto_objs)
                            objectives_input = "\n".join(f"- {x}" for x in auto_objs)
                        else:
                            user_list = [x.strip("- ").strip() for x in objs.splitlines() if x.strip()]
                            save_objectives(lid, user_list)
                            objectives_input = objs

                        with db_cursor(commit=True) as (_conn, c):
                            c.execute("UPDATE lectures SET slide_count=%s WHERE id=%s", (len(images), lid))

                        # CONDITIONALLY GENERATE CARDS
                        if gen_cards:
                            status.write("Generating Flashcards (high-yield gated)...")
                            for i in range(0, len(images), 10):
                                batch_imgs = images[i:i + 10]
                                batch_indices = list(range(i, min(i + 10, len(images))))
                                new_cards, error = generate_cards_hybrid(lid, batch_indices, batch_imgs, cache, objectives_input, i)

                                if error:
                                    status.write(f"AI flashcard note: {error}")

                                if new_cards:
                                    with db_cursor(commit=True) as (_conn, c):
                                        for f, b in new_cards:
                                            c.execute(
                                                "INSERT INTO cards (lecture_id, front, back, next_review) VALUES (%s,%s,%s,%s)",
                                                (lid, f, b, date.today()),
                                            )
                        else:
                            status.write("Skipping Flashcard Generation (Slides & PDF saved).")
                            time.sleep(1)

                    status.update(label="Complete!", state="complete", expanded=False)
                    st.success("Upload Finished!")
                    time.sleep(1)
                    st.rerun()

    with tab_manage:
        new_class = st.text_input("Create New Class Name")
        if st.button("Create Class"):
            if not new_class.strip():
                st.warning("Class name cannot be empty.")
            else:
                with db_cursor(commit=True) as (_conn, c):
                    c.execute("INSERT INTO classes (name) VALUES (%s)", (new_class.strip(),))
                st.rerun()

        st.write("---")
        st.write("🗑️ **Danger Zone**")

        with db_cursor() as (_conn, c):
            c.execute("SELECT id, name FROM exams")
            all_exams = c.fetchall()

        if all_exams:
            e_del_map = {n: i for i, n in all_exams}
            del_target = st.selectbox("Select Topic to Delete", list(e_del_map.keys()))
            if st.button("Permanently Delete Topic"):
                eid = e_del_map[del_target]

                with db_cursor() as (_conn, c):
                    c.execute("SELECT id, pdf_path FROM lectures WHERE exam_id=%s", (eid,))
                    l_rows = c.fetchall()

                # delete local cache folders
                for (lid, _pdf_path) in l_rows:
                    shutil.rmtree(os.path.join(SLIDE_DIR, str(lid)), ignore_errors=True)

                # Optional: attempt storage deletes (only if policies allow)
                deleted = 0
                for (_lid, pdf_path) in l_rows:
                    if pdf_path:
                        if delete_pdf_from_supabase(pdf_path):
                            deleted += 1

                with db_cursor(commit=True) as (_conn, c):
                    c.execute("DELETE FROM exams WHERE id=%s", (eid,))

                st.toast(f"Deleted topic. PDFs deleted: {deleted}/{len(l_rows)} (depends on storage policies).", icon="🧹")
                st.rerun()

# ------------------------------------------
# 3. ACTIVE LEARNING
# ------------------------------------------
elif nav == "Active Learning":
    st.title("👨‍🏫 Active Learning")

    with db_cursor() as (_conn, c):
        c.execute("SELECT l.id, l.name, e.name FROM lectures l JOIN exams e ON l.exam_id = e.id")
        all_lecs = c.fetchall()

    if not all_lecs:
        st.info("No lectures found.")
    else:
        l_ids = [l[0] for l in all_lecs]
        l_labels = [f"{l[1]} ({l[2]})" for l in all_lecs]
        default_idx = (
            l_ids.index(st.session_state.active_lecture_id)
            if "active_lecture_id" in st.session_state and st.session_state.active_lecture_id in l_ids
            else 0
        )
        sel_label = st.selectbox("Current Lecture", l_labels, index=default_idx)
        lid = l_ids[l_labels.index(sel_label)]
        lec_path = os.path.join(SLIDE_DIR, str(lid))

        try:
            # FIX: Capture the returned cache directly
            cache = ensure_local_assets_for_lecture(lid)
        except Exception as e:
            st.error(str(e))
            st.stop()

        # REMOVED: cache = load_slide_cache(lid) 
        # We don't need to load from disk again; ensure_local_assets returned it.

        if not cache:
            # This now only triggers if the PDF truly had no content to cache
            st.error("No slides data found (cache is empty). Try re-uploading the lecture.")
            st.stop()

        slides = sorted(
            [f for f in os.listdir(lec_path) if f.endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        
        start = st.session_state.read_idx

        col_slides, col_tools = st.columns([10, 5])
        with col_slides:
            end = min(start + 5, len(slides))
            st.caption(f"Slides {start+1}-{end} of {len(slides)}")
            st.progress(end / len(slides))

            current_images = []
            current_indices = list(range(start, end))
            for i in range(start, end):
                img = PIL.Image.open(os.path.join(lec_path, slides[i]))
                current_images.append(img)
                st.image(img, use_container_width=True, output_format="PNG")

            c_prev, c_next = st.columns(2)
            if c_prev.button("⬅️ Previous", use_container_width=True) and start > 0:
                st.session_state.read_idx = max(0, start - 5)
                st.session_state.quiz_data = None
                st.rerun()
            if c_next.button("Next ➡️", use_container_width=True) and end < len(slides):
                st.session_state.read_idx = end
                st.session_state.quiz_data = None
                st.rerun()

        with col_tools:
            st.write("#### 🧠 Quick Quiz")
            if st.button("Generate Quiz", type="primary"):
                with st.spinner("AI thinking..."):
                    st.session_state.quiz_data = generate_quiz_hybrid(lid, current_indices, current_images, cache)

            if st.session_state.quiz_data:
                q_data = st.session_state.quiz_data
                if isinstance(q_data, list) and q_data and isinstance(q_data[0], dict) and "error" in q_data[0]:
                    st.error(f"AI Error: {q_data[0]['error']}")
                else:
                    for i, q in enumerate(q_data):
                        with st.expander(f"Q{i+1}: {q['question']}", expanded=True):
                            ans = st.radio("Select:", q["options"], key=f"q_{i}")
                            if st.button("Check", key=f"chk_{i}"):
                                corr = q["options"][q["correct_index"]]
                                if ans == corr:
                                    st.success("Correct!")
                                else:
                                    st.error(f"Wrong. Answer: {corr}")
                                    st.info(q["explanation"])

# ------------------------------------------
# 4. EDITOR
# ------------------------------------------
elif nav == "Editor":
    st.title("🛠️ Card Editor")

    with db_cursor() as (conn, c):
        c.execute("SELECT id, name FROM exams")
        exams = c.fetchall()

    if exams:
        e_map = {name: id for id, name in exams}
        filter_exam = st.selectbox("Topic", list(e_map.keys()))
        eid = e_map[filter_exam]

        with db_cursor() as (conn, _c):
            query = """
            SELECT c.id, c.front, c.back
            FROM cards c
            JOIN lectures l ON c.lecture_id = l.id
            WHERE l.exam_id = %s
            """
            df = pd.read_sql(query, conn, params=(eid,))

        edited = st.data_editor(df, num_rows="dynamic", key="editor", use_container_width=True)
        if st.button("Save Changes", type="primary"):
            with db_cursor(commit=True) as (_conn, c):
                for _, row in edited.iterrows():
                    c.execute(
                        "UPDATE cards SET front=%s, back=%s WHERE id=%s",
                        (row["front"], row["back"], int(row["id"])),
                    )
            st.toast("Saved successfully!", icon="✅")
    else:
        st.info("No topics found.")



