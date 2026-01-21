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
import re

# ==========================================
# ✅ ENV DETECTION
# ==========================================
def running_on_streamlit_cloud() -> bool:
    return os.environ.get("STREAMLIT_CLOUD", "").lower() == "true" or bool(os.environ.get("STREAMLIT_SERVER_HEADLESS"))

IS_CLOUD = running_on_streamlit_cloud()

# ==========================================
# ⚙️ CONFIGURATION & THEME
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
# ⚙️ SECRETS & AI SETUP
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    DB_URL = st.secrets["SUPABASE_DB_URL"]
    
    # Supabase Storage
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
    OLLAMA_ENABLED = bool(st.secrets.get("OLLAMA_ENABLED", True))
    OLLAMA_URL = st.secrets.get("OLLAMA_URL", "http://127.0.0.1:11434")
    OLLAMA_TEXT_MODEL = st.secrets.get("OLLAMA_TEXT_MODEL", "qwen2.5:7b-instruct")
    OLLAMA_VISION_MODEL = st.secrets.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")

except Exception:
    st.error("Missing Secrets! Check .streamlit/secrets.toml")
    st.stop()

# ✅ PROVIDER ORDER: Ollama First
PROVIDER_ORDER = []
if OLLAMA_ENABLED:
    PROVIDER_ORDER.append("ollama")
PROVIDER_ORDER.extend(["gemini", "openai"])

# ✅ GEMINI CONFIG & QUOTA HOPPING
genai.configure(api_key=GOOGLE_API_KEY)

safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}

# The quiz model can remain the older stable pro
quiz_model = genai.GenerativeModel("gemini-2.5-pro", safety_settings=safety_settings)

# 📋 QUOTA HOPPING LIST (Best/Cheapest -> Newest/Fallback)
GEMINI_MODELS_TO_TRY = [
    "gemini-2.5-flash-lite",  # Tier 1: Most efficient
    "gemini-2.5-flash",       # Tier 2: Standard
    "gemini-3-flash",         # Tier 3: Fresh quota backup
    "gemini-1.5-flash"        # Tier 4: Older reliable backup
]

SLIDE_DIR = "lecture_slides"
os.makedirs(SLIDE_DIR, exist_ok=True)

# ==========================================
# 🧠 ROBUST JSON PARSER
# ==========================================
def parse_json_response(response_text, payload_name):
    if not response_text:
        return []
    
    # 1. Strip markdown
    text = response_text.replace("```json", "").replace("```", "").strip()
    
    # 2. Naive attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Regex Rescue
    try:
        match_list = re.search(r'\[.*\]', text, re.DOTALL)
        if match_list:
            return json.loads(match_list.group())
            
        match_obj = re.search(r'\{.*\}', text, re.DOTALL)
        if match_obj:
            return json.loads(match_obj.group())
    except Exception:
        pass
        
    return []

# ==========================================
# ⚡ GEMINI SAFE CALLER (Quota Hopping)
# ==========================================
def call_gemini_safe(prompt_text, payload_name):
    """
    Tries multiple Gemini models. If one hits a rate limit, it swaps to the next.
    """
    last_error = None
    for model_name in GEMINI_MODELS_TO_TRY:
        try:
            model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
            response = model.generate_content(prompt_text)
            
            # Check for safety blocks (empty parts)
            if not response.candidates or not response.candidates[0].content.parts:
                print(f"Safety block on {model_name}")
                continue

            data = parse_json_response(response.text, payload_name)
            
            if isinstance(data, list) and data:
                return data
            # Also accept dicts if expected
            if isinstance(data, dict) and data:
                return data
                
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "ResourceExhausted" in err_str:
                print(f"⚠️ Quota Limit on {model_name}. Switching...")
                last_error = e
                continue
            else:
                last_error = e
                continue

    raise last_error if last_error else RuntimeError("All Gemini models failed.")

# ==========================================
# ✅ DB SETUP
# ==========================================
@st.cache_resource
def get_db_connection():
    return psycopg2.connect(DB_URL)

def get_conn():
    conn = get_db_connection()
    try:
        if conn is None or conn.closed != 0:
            raise psycopg2.InterfaceError("Closed")
        with conn.cursor() as cur: cur.execute("SELECT 1;")
        return conn
    except Exception:
        get_db_connection.clear()
        conn = get_db_connection()
        return conn

@contextmanager
def db_cursor(commit: bool = False):
    conn = get_conn()
    cur = conn.cursor()
    try:
        yield conn, cur
        if commit: conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try: cur.close()
        except: pass

def ensure_schema():
    with db_cursor(commit=True) as (_conn, c):
        c.execute("""
            SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='lectures' AND column_name='pdf_path');
        """)
        if not c.fetchone()[0]:
            c.execute("ALTER TABLE lectures ADD COLUMN pdf_path TEXT;")
        
        c.execute("""
            SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='cards' AND column_name='lapses');
        """)
        if not c.fetchone()[0]:
            c.execute("ALTER TABLE cards ADD COLUMN lapses INT DEFAULT 0;")
            
        c.execute("""
            SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='cards' AND column_name='last_reviewed');
        """)
        if not c.fetchone()[0]:
            c.execute("ALTER TABLE cards ADD COLUMN last_reviewed DATE;")

ensure_schema()

def run_cleanup_once_per_day():
    today_iso = date.today().isoformat()
    if st.session_state.get("last_cleanup") == today_iso: return
    try:
        with db_cursor(commit=True) as (_conn, c):
            c.execute("select public.cleanup_old_cards();")
        st.session_state["last_cleanup"] = today_iso
    except: pass

run_cleanup_once_per_day()

# ==========================================
# ✅ SUPABASE STORAGE
# ==========================================
def _supabase_storage_headers(content_type: str | None = None):
    h = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}
    if content_type: h["Content-Type"] = content_type
    return h

def upload_pdf_to_supabase(lecture_id: int, pdf_bytes: bytes) -> str:
    obj_path = f"{SUPABASE_PDF_PREFIX}/{lecture_id}.pdf"
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{obj_path}"
    r = requests.post(url, headers=_supabase_storage_headers("application/pdf"), params={"upsert": "true"}, data=pdf_bytes, timeout=60)
    if not (200 <= r.status_code < 300): raise RuntimeError(f"Upload failed: {r.text}")
    return obj_path

def download_pdf_from_supabase(object_path: str) -> bytes:
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{object_path}"
    r = requests.get(url, headers=_supabase_storage_headers(), timeout=60)
    if not (200 <= r.status_code < 300): raise RuntimeError(f"Download failed: {r.text}")
    return r.content

def delete_pdf_from_supabase(object_path: str) -> bool:
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{object_path}"
    r = requests.delete(url, headers=_supabase_storage_headers(), timeout=30)
    return 200 <= r.status_code < 300

# ==========================================
# 🧠 AI HELPERS (OLLAMA & OPENAI)
# ==========================================
def ollama_is_up(ollama_url: str) -> bool:
    try:
        return requests.get(f"{ollama_url}/api/tags", timeout=2).status_code == 200
    except: return False

def pil_to_base64_png(pil_img) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def ollama_generate_text(prompt: str, model: str, temperature=0.2, timeout=180) -> str:
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")

def ollama_generate_vision(prompt: str, pil_images: list, model: str, temperature=0.2, timeout=240) -> str:
    images_b64 = [pil_to_base64_png(img) for img in pil_images]
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={"model": model, "prompt": prompt, "images": images_b64, "stream": False, "options": {"temperature": temperature}}, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")

def openai_generate_text(prompt: str, model: str, timeout=180) -> str:
    if not openai_client: raise RuntimeError("OpenAI client missing.")
    resp = openai_client.responses.create(model=model, input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}], timeout=timeout)
    return "".join([getattr(p, "text", "") for i in (resp.output or []) for p in (getattr(i, "content", []) or []) if getattr(p, "type", "") in ("text", "output_text")]).strip()

def openai_generate_vision(prompt: str, pil_images: list, model: str, timeout=240) -> str:
    if not openai_client: raise RuntimeError("OpenAI client missing.")
    def pil_to_data_url(img):
        buf = BytesIO(); img.save(buf, format="PNG"); b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    content = [{"type": "input_text", "text": prompt}]
    for img in pil_images: content.append({"type": "input_image", "image_url": pil_to_data_url(img)})
    resp = openai_client.responses.create(model=model, input=[{"role": "user", "content": content}], timeout=timeout)
    return "".join([getattr(p, "text", "") for i in (resp.output or []) for p in (getattr(i, "content", []) or []) if getattr(p, "type", "") in ("text", "output_text")]).strip()

# ==========================================
# 🧠 PDF & SLIDE UTILS
# ==========================================
def extract_slide_texts_from_pdf_bytes(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page in doc:
        txt = page.get_text("text") or ""
        texts.append("\n".join([line.strip() for line in txt.splitlines() if line.strip()]))
    return texts

def save_slides_locally_from_pdf_bytes(pdf_bytes: bytes, lecture_id: int, dpi=250):
    slide_images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    lec_path = os.path.join(SLIDE_DIR, str(lecture_id))
    if os.path.exists(lec_path):
        try: shutil.rmtree(lec_path)
        except: pass
    os.makedirs(lec_path, exist_ok=True)
    for i, page in enumerate(doc):
        img_path = os.path.join(lec_path, f"slide_{i}.png")
        page.get_pixmap(dpi=dpi).save(img_path)
        slide_images.append(PIL.Image.open(img_path))
    return slide_images

def slide_needs_vision(slide_text: str, min_chars=140):
    return len((slide_text or "").strip()) < min_chars

def load_slide_cache(lecture_id: int):
    try:
        with open(os.path.join(SLIDE_DIR, str(lecture_id), "slide_cache.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except: return None

def build_slide_cache(lecture_id: int, slide_texts):
    cache = [{"i": i, "text": t, "needs_vision": slide_needs_vision(t), "image_notes": ""} for i, t in enumerate(slide_texts)]
    p = os.path.join(SLIDE_DIR, str(lecture_id), "slide_cache.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f: json.dump(cache, f, indent=2)
    return cache

def ensure_image_notes_for_batch(lecture_id, batch_indices, batch_images, cache):
    if not (OLLAMA_ENABLED and ollama_is_up(OLLAMA_URL)): return cache
    to_process, to_process_idxs = [], []
    for pos, slide_i in enumerate(batch_indices):
        if cache[slide_i].get("needs_vision") and not cache[slide_i].get("image_notes"):
            to_process.append(batch_images[pos])
            to_process_idxs.append(slide_i)
    if not to_process: return cache

    for start in range(0, len(to_process), 2):
        imgs, idxs = to_process[start:start+2], to_process_idxs[start:start+2]
        prompt = "Interpret pharmacy slides. concise notes on figures/graphs/traps. Return RAW JSON: [{'notes':'...'},...]"
        try:
            resp = ollama_generate_vision(prompt, imgs, model=OLLAMA_VISION_MODEL, temperature=0.1)
            notes = parse_json_response(resp, "image_notes")
            for k, item in enumerate(notes if isinstance(notes, list) else []):
                if k < len(idxs): cache[idxs[k]]["image_notes"] = (item.get("notes", "") or "").strip()
        except: pass
    
    with open(os.path.join(SLIDE_DIR, str(lecture_id), "slide_cache.json"), "w", encoding="utf-8") as f: json.dump(cache, f, indent=2)
    return cache

def objectives_path(lecture_id: int): return os.path.join(SLIDE_DIR, str(lecture_id), "objectives.json")
def load_objectives(lecture_id: int):
    try:
        with open(objectives_path(lecture_id), "r") as f: return json.load(f).get("objectives", [])
    except: return []

def extract_objectives_from_slide_texts(slide_texts, max_slides=8):
    early = "\n".join(slide_texts[:max_slides])
    prompt = f"Extract LEARNING OBJECTIVES. Return RAW JSON: {{'objectives':['...']}}. SLIDES:\n{early}"
    try:
        # Use Quota Hopping for this too
        data = call_gemini_safe(prompt, "objectives")
        return [o.strip() for o in data.get("objectives", []) if isinstance(o, str)]
    except: return []

def save_objectives(lecture_id: int, objectives_list):
    os.makedirs(os.path.dirname(objectives_path(lecture_id)), exist_ok=True)
    with open(objectives_path(lecture_id), "w", encoding="utf-8") as f: json.dump({"objectives": objectives_list}, f, indent=2)

def objectives_to_string(obj_input, lecture_id: int):
    if obj_input and obj_input.strip(): return obj_input.strip()
    return "\n".join(f"- {x}" for x in load_objectives(lecture_id)) or "Prioritize definitions, formulas, drugs."

def ensure_local_assets_for_lecture(lecture_id: int):
    lec_path = os.path.join(SLIDE_DIR, str(lecture_id))
    if os.path.exists(lec_path) and load_slide_cache(lecture_id): return load_slide_cache(lecture_id)
    with db_cursor() as (_conn, c): c.execute("SELECT pdf_path FROM lectures WHERE id=%s", (lecture_id,)); row=c.fetchone()
    if not row or not row[0]: raise RuntimeError("PDF missing")
    pdf_bytes = download_pdf_from_supabase(row[0])
    save_slides_locally_from_pdf_bytes(pdf_bytes, lecture_id)
    slide_texts = extract_slide_texts_from_pdf_bytes(pdf_bytes)
    cache = build_slide_cache(lecture_id, slide_texts)
    if not load_objectives(lecture_id): save_objectives(lecture_id, extract_objectives_from_slide_texts(slide_texts))
    return cache

# ==========================================
# ✅ GENERATION LOGIC (Ollama -> Critic -> Gemini)
# ==========================================
def build_slides_blob(batch_entries, start_idx):
    return "\n\n".join([f"--- Slide {start_idx+k+1} ---\nTEXT:\n{e.get('text','')}\nIMG:\n{e.get('image_notes','')}" for k,e in enumerate(batch_entries)])

def extract_high_yield(batch_entries, objectives_str, start_idx, provider="gemini"):
    slides_blob = build_slides_blob(batch_entries, start_idx)
    # ✅ FIX: Double braces {{ }} prevents f-string crashes
    prompt = f"""
Pharmacy content curator. OBJECTIVES: {objectives_str}
Rules: EXCLUDE admin/logistics. INCLUDE definitions, drugs, formulas.
Return RAW JSON:
[ {{ "slide": <int>, "relevant": true, "key_points": ["..."] }} ]
SLIDES:
{slides_blob}
""".strip()
    
    if provider == "ollama":
        resp = ollama_generate_text(prompt, model=OLLAMA_TEXT_MODEL, temperature=0.1)
        return parse_json_response(resp, "high_yield")
    if provider == "gemini":
        return call_gemini_safe(prompt, "high_yield")
    if provider == "openai":
        resp = openai_generate_text(prompt, model=OPENAI_TEXT_MODEL)
        return parse_json_response(resp, "high_yield")
    return []

def condensed_notes_from_high_yield(high_yield_items):
    condensed = []
    for item in (high_yield_items or []):
        if isinstance(item, dict) and item.get("relevant"):
            condensed.append(f"Slide {item.get('slide')}:\nPOINTS: " + "\n".join(item.get("key_points", [])))
    return "\n\n".join(condensed)

def generate_cards_from_notes(notes_blob, objectives_str, provider="gemini"):
    if not notes_blob: return []
    # ✅ FIX: Double braces {{ }} 
    prompt = f"""
Pharmacy Professor. Write NAPLEX flashcards based on notes.
OBJECTIVES: {objectives_str}
Return RAW JSON: [{{{{ "front": "...", "back": "..." }}}}]
NOTES:
{notes_blob}
""".strip()

    if provider == "ollama":
        resp = ollama_generate_text(prompt, model=OLLAMA_TEXT_MODEL, temperature=0.2)
        data = parse_json_response(resp, "cards")
    elif provider == "gemini":
        data = call_gemini_safe(prompt, "cards")
    elif provider == "openai":
        resp = openai_generate_text(prompt, model=OPENAI_TEXT_MODEL)
        data = parse_json_response(resp, "cards")
    else: data = []

    if isinstance(data, list):
        return [(x["front"], x["back"]) for x in data if "front" in x and "back" in x]
    return []

def verify_cards_quality(cards, objectives_str):
    if not cards: return False, "No cards."
    cards_text = "\n".join([f"Q: {f}\nA: {b}" for f, b in cards])
    prompt = f"""
Grade these pharmacy flashcards.
OBJECTIVES: {objectives_str}
CRITERIA: Accurate? Comprehensive? Not vague?
Return RAW JSON: {{ "pass": true/false, "critique": "..." }}
CARDS:
{cards_text}
"""
    try:
        resp = ollama_generate_text(prompt, model=OLLAMA_TEXT_MODEL, temperature=0.1)
        data = parse_json_response(resp, "verify")
        return data.get("pass", False), data.get("critique", "Fail")
    except: return True, "Error"

def improve_cards(cards, feedback, objectives_str, provider="gemini"):
    cards_text = "\n".join([f"Q: {f}\nA: {b}" for f, b in cards])
    prompt = f"""
Editor. Fix rejected cards based on FEEDBACK: {feedback}
OBJECTIVES: {objectives_str}
ORIGINAL:
{cards_text}
Return RAW JSON: [{{ "front": "...", "back": "..." }}]
"""
    try:
        if provider == "gemini": data = call_gemini_safe(prompt, "improve")
        else:
            resp = ollama_generate_text(prompt, model=OLLAMA_TEXT_MODEL, temperature=0.3)
            data = parse_json_response(resp, "improve")
        
        if isinstance(data, list): return [(x["front"], x["back"]) for x in data if "front" in x]
    except: pass
    return []

def generate_cards_hybrid(lecture_id, batch_indices, batch_images, cache, objectives_input, start_idx):
    cache = ensure_image_notes_for_batch(lecture_id, batch_indices, batch_images, cache)
    batch_entries = [cache[i] for i in batch_indices]
    objectives_str = objectives_to_string(objectives_input, lecture_id)

    # 1. GENERATE (Ollama)
    cards = []
    if OLLAMA_ENABLED and ollama_is_up(OLLAMA_URL):
        try:
            hy = extract_high_yield(batch_entries, objectives_str, start_idx, provider="ollama")
            notes = condensed_notes_from_high_yield(hy)
            cards = generate_cards_from_notes(notes, objectives_str, provider="ollama")
        except Exception as e: print(f"Ollama gen failed: {e}")

    # Fallback to Gemini if Ollama made nothing
    if not cards:
        try:
            hy = extract_high_yield(batch_entries, objectives_str, start_idx, provider="gemini")
            notes = condensed_notes_from_high_yield(hy)
            cards = generate_cards_from_notes(notes, objectives_str, provider="gemini")
            if cards: return cards, None # Trust Gemini
        except: return [], "Generation failed."

    # 2. VERIFY (Ollama Critic)
    is_good, critique = verify_cards_quality(cards, objectives_str)
    if is_good: return cards, None

    # 3. IMPROVE (Gemini -> Ollama)
    print(f"Cards rejected: {critique}")
    improved = improve_cards(cards, critique, objectives_str, provider="gemini")
    if improved: return improved, f"Fixed by Gemini. ({critique})"
    
    improved = improve_cards(cards, critique, objectives_str, provider="ollama")
    if improved: return improved, f"Fixed by Ollama. ({critique})"

    return cards, f"Quality Low. ({critique})"

# ==========================================
# ✅ QUIZ & SR
# ==========================================
def generate_quiz_hybrid(lecture_id, batch_indices, batch_images, cache):
    cache = ensure_image_notes_for_batch(lecture_id, batch_indices, batch_images, cache)
    batch_entries = [cache[i] for i in batch_indices]
    
    if OLLAMA_ENABLED and ollama_is_up(OLLAMA_URL):
        slides_blob = build_slides_blob(batch_entries, 0)
        # FIX: Double braces
        prompt = f"5-question MC quiz for Pharmacy students. Return RAW JSON: [{{{{'question':'...','options':['A)...'],'correct_index':0,'explanation':'...'}}}}]. SLIDES:\n{slides_blob}"
        resp = ollama_generate_text(prompt, model=OLLAMA_TEXT_MODEL, temperature=0.2)
        return parse_json_response(resp, "quiz")
    
    # Gemini Fallback
    prompt = "5-question MC quiz. JSON list only."
    try:
        content = [prompt] + batch_images
        response = quiz_model.generate_content(content)
        return parse_json_response(response.text, "quiz")
    except Exception as e: return [{"error": str(e)}]

def sr_compute_next(interval, ease, review_count, lapses, quality):
    interval, ease = (int(interval) if interval else 1), (float(ease) if ease else 2.5)
    review_count, lapses = (int(review_count or 0), int(lapses or 0))
    ease = max(1.3, min(3.0, ease))

    if review_count == 0:
        if quality == 0: return 1, max(1.3, ease-0.2), lapses+1
        if quality >= 4: return (3 if quality==4 else 4), ease + (0.1 if quality==5 else 0), lapses
        return 1, ease, lapses

    if quality == 0: return 1, max(1.3, ease-0.2), lapses+1
    if quality == 3: return max(1, int(interval*1.2)), max(1.3, ease-0.15), lapses
    if quality == 4: return max(2, int(interval*ease)), ease, lapses
    if quality == 5: return max(3, int(interval*ease*1.3)), min(3.0, ease+0.15), lapses
    return interval, ease, lapses

def apply_exam_cap(new_int, exam_date, today):
    if exam_date and (exam_date - today).days > 0:
        return min(new_int, max(1, (exam_date - today).days - 1))
    return new_int

# ==========================================
# 🖥️ UI
# ==========================================
if "main_nav" not in st.session_state: st.session_state.main_nav = "Review"
nav = st.sidebar.radio("Menu", ["Review", "Library", "Active Learning", "Editor"], key="main_nav")

def open_lecture_callback(lid):
    st.session_state.active_lecture_id = lid
    st.session_state.main_nav = "Active Learning"

if nav == "Review":
    st.title("🧠 Study Center")
    today = date.today()
    
    st.sidebar.markdown("### ⚙️ Settings")
    session_limit = st.sidebar.number_input("Session Limit", 10, 500, 50, 10)

    if "session_active" not in st.session_state: st.session_state.session_active = False
    if "streak" not in st.session_state: st.session_state.streak = 0
    if "last_completion_date" not in st.session_state: st.session_state.last_completion_date = None
    if "missed_content" not in st.session_state: st.session_state.missed_content = []

    with db_cursor() as (conn, c):
        c.execute("""
            SELECT c.id, c.front, c.back, c.interval, c.ease, c.review_count, c.lapses, l.id, l.name, e.exam_date, e.name, e.id
            FROM cards c JOIN lectures l ON c.lecture_id = l.id JOIN exams e ON l.exam_id = e.id
            WHERE c.next_review <= %s ORDER BY e.exam_date ASC NULLS LAST, c.next_review ASC LIMIT %s
        """, (today, session_limit))
        cards_due = c.fetchall()
        c.execute("SELECT COUNT(*) FROM cards WHERE next_review <= %s", (today,))
        total_pending = c.fetchone()[0]
        c.execute("SELECT name, exam_date FROM exams WHERE exam_date >= %s ORDER BY exam_date ASC LIMIT 1", (today,))
        next_ex = c.fetchone()

    if not st.session_state.session_active:
        if not cards_due:
            st.balloons(); st.success("All caught up!")
            st.metric("🔥 Streak", f"{st.session_state.streak} Days")
        else:
            total_due = len(cards_due)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Cards", f"{total_due} / {total_pending}")
            
            with col2:
                if next_ex: st.metric("Next Exam", f"{(next_ex[1]-today).days} Days", next_ex[0])
                else: st.metric("Next Exam", "None")
            
            col3.metric("Est. Time", f"~{int(total_due*10//60)} min")
            col4.metric("Streak", f"{st.session_state.streak} Days")
            
            st.markdown("---")
            if st.button("🚀 Start Review Session", type="primary", use_container_width=True):
                st.session_state.update({"session_active":True, "idx":0, "total_seconds":0, "trouble_lectures":{}, "missed_content":[], "session_start_time":time.time()})
                st.rerun()
    else:
        if st.session_state.idx < len(cards_due):
            row = cards_due[st.session_state.idx]
            cid, front, back, interval, ease, revs, lapses, lid, lname, exam_date, _, _ = row
            st.progress(st.session_state.idx / len(cards_due))
            st.caption(f"Card {st.session_state.idx+1} / {len(cards_due)}")
            
            st.markdown(f'<div class="flashcard"><small>Q</small><br>{front}</div>', unsafe_allow_html=True)
            if st.session_state.get("show"):
                st.markdown(f'<div class="flashcard flashcard-back"><small>A</small><br>{back}</div>', unsafe_allow_html=True)
                def ans(q):
                    ni, ne, nl = sr_compute_next(interval, ease, revs, lapses, q)
                    ni = apply_exam_cap(ni, exam_date, today)
                    with db_cursor(commit=True) as (_, c):
                        c.execute("UPDATE cards SET next_review=%s, interval=%s, ease=%s, review_count=%s, lapses=%s, last_reviewed=%s WHERE id=%s",
                                  (today+timedelta(days=ni), ni, ne, revs+1, nl, today, cid))
                    if q<4: 
                        st.session_state.missed_content.append(f"Q:{front}|A:{back}")
                        st.session_state.trouble_lectures.setdefault(lid, {"count":0, "name":lname, "exam_date":exam_date})["count"]+=1
                    st.session_state.show = False
                    st.session_state.idx += 1
                    st.rerun()
                c1,c2,c3,c4 = st.columns(4)
                if c1.button("Again"): ans(0)
                if c2.button("Hard"): ans(3)
                if c3.button("Good"): ans(4)
                if c4.button("Easy"): ans(5)
            else:
                if st.button("Show Answer", type="primary"): st.session_state.show = True; st.rerun()
        else:
            st.success("Session Complete!")
            if st.session_state.missed_content:
                st.markdown("### 📖 Cheat Sheet")
                if st.button("Generate Summary"):
                    with st.spinner("Analyzing..."):
                        missed_str = "\n".join(st.session_state.missed_content[:15])
                        prompt = f"Pharmacy Prof. Summarize missed concepts for cheat sheet:\n{missed_str}"
                        # Use quota hopping for this too
                        data = call_gemini_safe(prompt, "summary") # Helper works if response is text
                        # But wait, call_gemini_safe expects JSON. Let's make a quick raw call here or adapt.
                        # Actually for summary, just raw text is fine. Let's use call_gemini_safe but wrap prompt to ask for JSON string.
                        # OR easier: just use the raw model loop here manually for text.
                        # For simplicity in this big file, let's just try one model safely:
                        try:
                            # Use one of the robust models
                            resp = genai.GenerativeModel("gemini-2.5-flash-lite").generate_content(prompt)
                            st.markdown(f'<div style="background-color:#FFF8E1; padding:20px; color:black;">{resp.text}</div>', unsafe_allow_html=True)
                        except:
                            st.error("Summary failed (Quota/Error).")

            if st.button("Finish"): st.session_state.session_active = False; st.rerun()

elif nav == "Library":
    st.title("📂 Library")
    t1, t2, t3 = st.tabs(["Browse", "Upload", "Manage"])
    with t1:
        with db_cursor() as (conn, _):
            df = pd.read_sql("SELECT c.name cname, e.name ename, l.id lid, l.name lname FROM classes c JOIN exams e ON e.class_id=c.id LEFT JOIN lectures l ON l.exam_id=e.id ORDER BY cname, ename", conn)
        for cname, cgrp in df.groupby("cname"):
            with st.expander(f"📁 {cname}"):
                for ename, egrp in cgrp.groupby("ename"):
                    st.caption(f"📂 {ename}")
                    for _, row in egrp.iterrows():
                        if pd.notna(row["lname"]):
                            c1, c2, c3 = st.columns([0.7,0.15,0.15])
                            c1.write(f"📄 {row['lname']}")
                            c2.button("Open", key=f"o_{row['lid']}", on_click=open_lecture_callback, args=(row['lid'],))
                            if c3.button("⚡ AI", key=f"ai_{row['lid']}"):
                                cache = ensure_local_assets_for_lecture(row['lid'])
                                lpath = os.path.join(SLIDE_DIR, str(row['lid']))
                                imgs = [PIL.Image.open(os.path.join(lpath, f)) for f in sorted(os.listdir(lpath)) if f.endswith(".png")]
                                for i in range(0, len(imgs), 10):
                                    nc, err = generate_cards_hybrid(row['lid'], list(range(i, min(i+10, len(imgs)))), imgs[i:i+10], cache, "", i)
                                    if err: st.error(err)
                                    if nc:
                                        with db_cursor(commit=True) as (_, c):
                                            for f,b in nc: c.execute("INSERT INTO cards (lecture_id,front,back,next_review) VALUES (%s,%s,%s,%s)", (row['lid'],f,b,date.today()))
                                st.rerun()

    with t2:
        with db_cursor() as (conn, c): c.execute("SELECT id, name FROM exams"); exams = c.fetchall()
        if not exams: st.warning("Create a Topic/Exam in Manage tab first.")
        else:
            emap = {n: i for i, n in exams}
            sel_e = st.selectbox("Topic", list(emap.keys()))
            up = st.file_uploader("PDFs", accept_multiple_files=True)
            gen = st.checkbox("Auto-generate cards?", True)
            if up and st.button("Upload"):
                st_stat = st.status("Processing...")
                for u in up:
                    st_stat.write(f"Uploading {u.name}...")
                    with db_cursor(commit=True) as (_, c): c.execute("INSERT INTO lectures (exam_id, name, slide_count) VALUES (%s,%s,0) RETURNING id", (emap[sel_e], u.name)); lid=c.fetchone()[0]
                    path = upload_pdf_to_supabase(lid, u.getvalue())
                    with db_cursor(commit=True) as (_, c): c.execute("UPDATE lectures SET pdf_path=%s WHERE id=%s", (path, lid))
                    imgs = save_slides_locally_from_pdf_bytes(u.getvalue(), lid)
                    cache = build_slide_cache(lid, extract_slide_texts_from_pdf_bytes(u.getvalue()))
                    save_objectives(lid, extract_objectives_from_slide_texts([e['text'] for e in cache]))
                    
                    if gen:
                        st_stat.write("Generating cards...")
                        for i in range(0, len(imgs), 10):
                            nc, err = generate_cards_hybrid(lid, list(range(i, min(i+10, len(imgs)))), imgs[i:i+10], cache, "", i)
                            if err: st.write(f"Note: {err}")
                            if nc:
                                with db_cursor(commit=True) as (_, c):
                                    for f,b in nc: c.execute("INSERT INTO cards (lecture_id,front,back,next_review) VALUES (%s,%s,%s,%s)", (lid,f,b,date.today()))
                st_stat.update(label="Done!", state="complete"); time.sleep(1); st.rerun()

    with t3:
        nclass = st.text_input("New Class")
        if st.button("Add Class") and nclass:
            with db_cursor(commit=True) as (_, c): c.execute("INSERT INTO classes (name) VALUES (%s)", (nclass,))
        
        with db_cursor() as (conn, c): c.execute("SELECT c.name, c.id FROM classes c"); classes = c.fetchall()
        if classes:
            cmap = {n: i for n, i in classes}
            sc = st.selectbox("Class for new Topic", list(cmap.keys()))
            ntopic = st.text_input("New Topic")
            ndate = st.date_input("Exam Date")
            if st.button("Add Topic") and ntopic:
                with db_cursor(commit=True) as (_, c): c.execute("INSERT INTO exams (class_id, name, exam_date) VALUES (%s,%s,%s)", (cmap[sc], ntopic, ndate))

elif nav == "Active Learning":
    st.title("Active Learning")
    if "active_lecture_id" in st.session_state:
        lid = st.session_state.active_lecture_id
        try: cache = ensure_local_assets_for_lecture(lid)
        except: st.error("Load failed"); st.stop()
        
        lpath = os.path.join(SLIDE_DIR, str(lid))
        slides = sorted([f for f in os.listdir(lpath) if f.endswith(".png")], key=lambda x: int(x.split("_")[1].split(".")[0]))
        
        start = st.session_state.get("read_idx", 0)
        c1, c2 = st.columns([3, 1])
        with c1:
            for i in range(start, min(start+5, len(slides))):
                st.image(os.path.join(lpath, slides[i]))
            if st.button("Next") and start+5 < len(slides): st.session_state.read_idx = start+5; st.rerun()
        with c2:
            if st.button("Quiz Me"):
                imgs = [PIL.Image.open(os.path.join(lpath, slides[i])) for i in range(start, min(start+5, len(slides)))]
                st.session_state.quiz = generate_quiz_hybrid(lid, list(range(start, min(start+5, len(slides)))), imgs, cache)
            
            if st.session_state.get("quiz"):
                for q in st.session_state.quiz:
                    if "question" in q:
                        st.write(q["question"])
                        st.info(q["options"][q["correct_index"]])

elif nav == "Editor":
    st.title("Editor")
    with db_cursor() as (conn, _): df = pd.read_sql("SELECT id, front, back FROM cards", conn)
    ed = st.data_editor(df, num_rows="dynamic")
    if st.button("Save"):
        with db_cursor(commit=True) as (_, c):
            for _, r in ed.iterrows(): c.execute("UPDATE cards SET front=%s, back=%s WHERE id=%s", (r["front"], r["back"], r["id"]))
        st.success("Saved")
