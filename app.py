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

# ==========================================
# ⚙️ CONFIGURATION & THEME ENGINE
# ==========================================
st.set_page_config(page_title="PharmPilot", page_icon="💊", layout="centered")

# --- THEME CSS DEFINITIONS ---
LIGHT_THEME = """
<style>
    /* Global Text */
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div, label { color: #000000 !important; }
    /* Backgrounds */
    .stApp { background-color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #F0F2F6; border-right: 1px solid #E6E6E6; }
    /* Inputs */
    input, textarea, select, .stSelectbox div[data-baseweb="select"] > div { 
        background-color: #FFFFFF !important; color: #000000 !important; 
    }
    /* Components */
    .flashcard { background-color: white; border: 1px solid #E0E0E0; border-left: 6px solid #4F8BF9; color: #000000 !important; }
    .flashcard-back { background-color: #eef6ff; border-left: 6px solid #00c853; color: #000000 !important; }
    div[data-testid="stExpander"] { background-color: #FFFFFF !important; border: 1px solid #E0E0E0; }
</style>
"""

DARK_THEME = """
<style>
    /* Global Text */
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div, label { color: #E0E0E0 !important; }
    /* Backgrounds */
    .stApp { background-color: #0E1117; }
    [data-testid="stSidebar"] { background-color: #262730; border-right: 1px solid #333; }
    /* Inputs */
    input, textarea, select { background-color: #1E1E1E !important; color: #E0E0E0 !important; }
    .stSelectbox div[data-baseweb="select"] > div { background-color: #1E1E1E !important; color: #E0E0E0 !important; }
    /* Components */
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
    /* Fix Tab Colors */
    button[data-baseweb="tab"] { background-color: transparent !important; }
</style>
"""

# SHARED CSS (Layouts that apply to both)
SHARED_CSS = """
<style>
    /* Button Exceptions (Keep text color inherent) */
    button p, button div { color: inherit !important; }
    /* Button Pop */
    .stButton>button { border-radius: 8px; height: 3em; font-weight: 600; border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.2); transition: transform 0.1s; }
    .stButton>button:active { transform: scale(0.98); }
</style>
"""

# --- SIDEBAR TOGGLE ---
st.sidebar.title("💊 PharmPilot")
dark_mode = st.sidebar.toggle("🌗 Dark Mode", value=True) # Default to Dark Mode since you liked it

# INJECT CSS BASED ON TOGGLE
if dark_mode:
    st.markdown(DARK_THEME + SHARED_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_THEME + SHARED_CSS, unsafe_allow_html=True)

st.sidebar.markdown("---")

# ==========================================
# ⚙️ SECRETS & SETUP
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    DB_URL = st.secrets["SUPABASE_DB_URL"]
except Exception:
    st.error("Missing Secrets! Make sure .streamlit/secrets.toml exists.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

flash_model = genai.GenerativeModel('gemini-2.5-flash', safety_settings=safety_settings)
quiz_model = genai.GenerativeModel('gemini-2.5-pro', safety_settings=safety_settings)

SLIDE_DIR = "lecture_slides"
if not os.path.exists(SLIDE_DIR):
    os.makedirs(SLIDE_DIR)

@st.cache_resource
def get_db_connection():
    return psycopg2.connect(DB_URL)

conn = get_db_connection()
c = conn.cursor()

# ==========================================
# 🧠 AI LOGIC
# ==========================================
def save_slides_locally(uploaded_file, lecture_id):
    slide_images = []
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    lec_path = os.path.join(SLIDE_DIR, str(lecture_id))
    
    def on_rm_error(_func, path, _exc_info):
        os.chmod(path, stat.S_IWRITE)
        os.unlink(path)

    if os.path.exists(lec_path):
        try:
            shutil.rmtree(lec_path, onerror=on_rm_error)
        except Exception:
            pass

    if not os.path.exists(lec_path):
        os.makedirs(lec_path)
    
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        img_path = os.path.join(lec_path, f"slide_{i}.png")
        pix.save(img_path)
        slide_images.append(PIL.Image.open(img_path))
    return slide_images

def generate_cards_batch_json(images, objectives, start_idx):
    prompt = f"""
    You are a Pharmacy Professor. Review these {len(images)} slides (Slides {start_idx+1}-{start_idx+len(images)}).
    OBJECTIVES: {objectives}
    INSTRUCTIONS:
    1. SCOPE: Identify the most important exam-relevant facts.
    2. DENSITY: Create roughly 1 card per slide on average.
    3. FILTER: Skip Title slides and generic "Objectives" slides.
    IMPORTANT: Return a raw JSON list of objects. Do NOT use Markdown formatting.
    Structure: [ {{"front": "Question...", "back": "Answer..."}} ]
    """
    try:
        response = flash_model.generate_content([prompt] + images)
        data = parse_json_response(response.text, "flashcards")
        valid_cards = []
        for item in data:
            if 'front' in item and 'back' in item:
                valid_cards.append((item['front'], item['back']))
        return valid_cards, None
    except Exception as e:
        return [], f"JSON Error: {str(e)}"

def generate_interactive_quiz(images):
    prompt = """
    Create a 5-question multiple choice quiz based on these slides.
    Target Audience: Pharmacy Students (NAPLEX level).
    IMPORTANT: Return ONLY a JSON list. Do not use Markdown blocks.
    Structure: [ { "question": "...", "options": ["A) ...", "B) ..."], "correct_index": 0, "explanation": "..." } ]
    """
    try:
        content = [prompt] + images
        response = quiz_model.generate_content(content)
        return parse_json_response(response.text, "quiz")
    except Exception as e:
        return [{"error": str(e)}]

def parse_json_response(response_text, payload_name):
    clean_text = response_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        start = clean_text.find("[")
        end = clean_text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(clean_text[start:end + 1])
        raise ValueError(f"Unable to parse {payload_name} JSON response.")

# ==========================================
# 🖥️ UI LOGIC
# ==========================================
if 'main_nav' not in st.session_state:
    st.session_state.main_nav = "Review"
if 'show' not in st.session_state:
    st.session_state.show = False
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'read_idx' not in st.session_state:
    st.session_state.read_idx = 0

nav = st.sidebar.radio("Menu", 
    ["Review", "Library", "Active Learning", "Editor"],
    key="main_nav"
)

def open_lecture_callback(lid):
    st.session_state.active_lecture_id = lid
    st.session_state.main_nav = "Active Learning"

# ------------------------------------------
# 1. REVIEW DASHBOARD (v5.8 Study Buddy)
# ------------------------------------------
if nav == "Review":
    st.title("🧠 Study Center")
    today = date.today()

    # --- 1. INITIALIZATION ---
    if 'session_active' not in st.session_state:
        st.session_state.session_active = False
    if 'streak' not in st.session_state:
        st.session_state.streak = 0
    if 'last_completion_date' not in st.session_state:
        st.session_state.last_completion_date = None
    if 'missed_content' not in st.session_state:
        st.session_state.missed_content = []

    # --- 2. DATABASE FETCH ---
    c.execute("""
        SELECT c.id, c.front, c.back, c.interval, c.ease, c.review_count, l.id, l.name, e.exam_date, e.name, e.id
        FROM cards c 
        JOIN lectures l ON c.lecture_id = l.id 
        JOIN exams e ON l.exam_id = e.id
        WHERE c.next_review <= %s 
        ORDER BY c.next_review ASC LIMIT 50
    """, (today,))
    cards_due = c.fetchall()

    # --- PREP SCREEN ---
    if not st.session_state.session_active:
        if not cards_due:
            st.balloons()
            st.success("🎉 All caught up for today!")
            st.metric("🔥 Current Streak", f"{st.session_state.streak} Days")
        else:
            total_due = len(cards_due)
            c.execute("SELECT name, exam_date FROM exams WHERE exam_date >= %s ORDER BY exam_date ASC LIMIT 1", (today,))
            next_ex = c.fetchone()
            
            st.markdown("### 📊 Session Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Cards Due", total_due)
            with col2:
                if next_ex:
                    days_left = (next_ex[1] - today).days
                    st.metric("Next Exam", f"{days_left} Days", next_ex[0])
                else: st.metric("Next Exam", "None Set")
            with col3:
                avg_speed = 10 if 'global_avg_speed' not in st.session_state else st.session_state.global_avg_speed
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

    # --- ACTIVE SESSION ---
    else:
        total_cards = len(cards_due)
        if st.session_state.idx < total_cards:
            remaining = total_cards - st.session_state.idx
            if st.session_state.idx > 0:
                avg_speed = st.session_state.total_seconds / st.session_state.idx
                time_display = f"⏱️ {int((avg_speed * remaining)//60)}m {int((avg_speed * remaining)%60)}s left"
            else: time_display = "⏱️ Calculating..."

            st.progress(st.session_state.idx / total_cards)
            c_left, c_right = st.columns(2)
            c_left.caption(f"Card {st.session_state.idx + 1} of {total_cards}")
            c_right.markdown(f"<p style='text-align:right; font-weight:bold;'>{time_display}</p>", unsafe_allow_html=True)

            cid, front, back, interval, ease, revs, lid, lname, exam_date, exam_name, eid = cards_due[st.session_state.idx]
            if 'card_load_time' not in st.session_state or not st.session_state.show:
                st.session_state.card_load_time = time.time()

            st.markdown(f'<div class="flashcard"><small>QUESTION</small><br>{front}</div>', unsafe_allow_html=True)

            if st.session_state.show:
                st.markdown(f'<div class="flashcard flashcard-back"><small>ANSWER</small><br>{back}</div>', unsafe_allow_html=True)
                
                def answer(quality):
                    # Save misses for Cheat Sheet
                    if quality in [0, 3]: 
                        st.session_state.missed_content.append(f"Q: {front} | A: {back}")
                        if lid not in st.session_state.trouble_lectures:
                            st.session_state.trouble_lectures[lid] = {"count": 1, "name": lname, "exam_date": exam_date}
                        else: st.session_state.trouble_lectures[lid]["count"] += 1

                    st.session_state.total_seconds += (time.time() - st.session_state.card_load_time)
                    new_ease, new_interval = ease, interval
                    if quality == 0: new_interval, new_ease = 1, max(1.3, ease - 0.2)
                    elif quality == 3: new_interval, new_ease = max(1, int(interval * 1.2)), max(1.3, ease - 0.15)
                    elif quality == 4: new_interval = max(1, int(interval * ease))
                    elif quality == 5: new_interval, new_ease = max(1, int(interval * ease * 1.3)), min(3.0, ease + 0.15)
                    
                    if exam_date and (exam_date - today).days > 0:
                        days_limit = (exam_date - today).days
                        if new_interval >= days_limit: new_interval = max(1, days_limit - 1)
                    
                    c.execute("UPDATE cards SET next_review=%s, interval=%s, ease=%s, review_count=%s WHERE id=%s", 
                             (today + timedelta(days=new_interval), new_interval, new_ease, revs + 1, cid))
                    conn.commit()
                    st.session_state.show, st.session_state.idx = False, st.session_state.idx + 1
                    st.rerun()

                c1, c2, c3, c4 = st.columns(4)
                if c1.button("❌ Again", use_container_width=True): answer(0)
                if c2.button("😓 Hard", use_container_width=True): answer(3)
                if c3.button("✅ Good", use_container_width=True): answer(4)
                if c4.button("🚀 Easy", use_container_width=True): answer(5)
            else:
                if st.button("Show Answer", type="primary", use_container_width=True):
                    st.session_state.show = True; st.rerun()
        
        else:
            # --- SESSION COMPLETION & STUDY BUDDY ---
            st.balloons()
            total_time = time.time() - st.session_state.session_start_time
            st.success(f"Session Finished! Total time: {int(total_time//60)}m {int(total_time%60)}s")

            # Update Streak
            if st.session_state.last_completion_date == today - timedelta(days=1):
                st.session_state.streak += 1
            elif st.session_state.last_completion_date != today:
                st.session_state.streak = 1
            st.session_state.last_completion_date = today

            # --- SMART CHEAT SHEET ---
            if st.session_state.missed_content:
                st.markdown("### 📖 Smart Cheat Sheet")
                if st.button("🪄 Generate AI Summary of Missed Concepts"):
                    with st.spinner("Analyzing your weak spots..."):
                        missed_str = "\n".join(st.session_state.missed_content[:15]) # Limit to 15 cards
                        prompt = f"As a Pharmacy Professor, summarize these missed concepts into a one-page clinical cheat sheet. Use bullet points and focus on high-yield exam facts:\n{missed_str}"
                        response = flash_model.generate_content(prompt)
                        st.markdown(f'<div style="background-color:#FFF8E1; padding:20px; border-radius:10px; color:black;">{response.text}</div>', unsafe_allow_html=True)

            # Recommendations
            if st.session_state.trouble_lectures:
                urgent = [v for k, v in st.session_state.trouble_lectures.items() if v['exam_date'] and (v['exam_date'] - today).days <= 14]
                if urgent:
                    top_trouble = sorted(urgent, key=lambda x: x['count'], reverse=True)[0]
                    st.warning(f"💊 **Lecture Recommendation:** Review **{top_trouble['name']}**.")
                    if st.button(f"📖 Deep Dive Now", type="primary"):
                        for lid, data in st.session_state.trouble_lectures.items():
                            if data['name'] == top_trouble['name']:
                                open_lecture_callback(lid); st.rerun()

            if st.button("Finish & Back to Prep Room"):
                st.session_state.session_active = False; st.rerun()

# --- 2. LIBRARY ---
elif nav == "Library":
    st.title("📂 Library")
    tab_browse, tab_upload, tab_manage = st.tabs(["📚 Browse Materials", "☁️ Upload New", "⚙️ Manage"])
    
    with tab_browse:
        big_query = """SELECT c.name as class_name, e.name as exam_name, e.id as exam_id,
                       l.id as lecture_id, l.name as lecture_name, l.slide_count
                       FROM classes c JOIN exams e ON e.class_id = c.id
                       LEFT JOIN lectures l ON l.exam_id = e.id ORDER BY c.name, e.name, l.name"""
        df = pd.read_sql(big_query, conn)
        if df.empty: st.info("Library is empty.")
        else:
            for class_name, class_group in df.groupby("class_name"):
                with st.expander(f"📁 {class_name}", expanded=False):
                    for exam_name, exam_group in class_group.groupby("exam_name"):
                        st.caption(f"📂 {exam_name}")
                        for _, row in exam_group.iterrows():
                            if pd.notna(row['lecture_name']):
                                lid, lname, lcount = int(row['lecture_id']), row['lecture_name'], int(row['slide_count'])
                                c1, c2, c3 = st.columns([0.6, 0.2, 0.2])
                                with c1: st.markdown(f"**{lname}**")
                                with c2: st.button("Open", key=f"op_{lid}", on_click=open_lecture_callback, args=(lid,))
                                with c3:
                                    if st.button("⚡ AI", key=f"rt_{lid}", help="Retry AI"):
                                        lec_path = os.path.join(SLIDE_DIR, str(lid))
                                        if os.path.exists(lec_path):
                                            slides = sorted([f for f in os.listdir(lec_path) if f.endswith(".png")], 
                                                            key=lambda x: int(x.split('_')[1].split('.')[0]))
                                            images = [PIL.Image.open(os.path.join(lec_path, s)) for s in slides]
                                            st.toast(f"Processing {len(images)} slides...", icon="⚡")
                                            for i in range(0, len(images), 10):
                                                batch = images[i : i + 10]
                                                new_cards, error = generate_cards_batch_json(batch, "Review", i)
                                                if error:
                                                    st.error(f"AI flashcard error: {error}")
                                                if new_cards:
                                                    for f, b in new_cards:
                                                        c.execute("INSERT INTO cards (lecture_id, front, back, next_review) VALUES (%s,%s,%s,%s)", (lid, f, b, date.today()))
                                                    conn.commit()
                                            st.rerun()
                        st.divider()

    with tab_upload:
        c.execute("SELECT id, name FROM classes")
        classes = c.fetchall()
        if not classes: st.warning("Create a Class in 'Manage' tab first!")
        else:
            c_map = {n: i for i, n in classes}
            sel_c = st.selectbox("Select Class", list(c_map.keys()))
            c.execute("SELECT id, name FROM exams WHERE class_id=%s", (c_map[sel_c],))
            exams = c.fetchall()
            with st.expander("➕ Add New Topic"):
                new_topic = st.text_input("Topic Name")
                d_val = st.date_input("Exam Date")
                if st.button("Create Topic"):
                    if not new_topic.strip():
                        st.warning("Topic name cannot be empty.")
                    else:
                        c.execute("INSERT INTO exams (class_id, name, exam_date) VALUES (%s,%s,%s)", (c_map[sel_c], new_topic.strip(), d_val))
                        conn.commit(); st.rerun()
            if exams:
                e_map = {n: i for i, n in exams}
                sel_e = st.selectbox("Select Topic", list(e_map.keys()))
                uploaded_files = st.file_uploader("Drop PDFs Here", type="pdf", accept_multiple_files=True)
                objs = st.text_area("Learning Objectives")
                if uploaded_files and st.button("🚀 Upload & Process", type="primary"):
                    status = st.status("Processing...", expanded=True)
                    for idx, uploaded in enumerate(uploaded_files):
                        status.write(f"Reading {uploaded.name}...")
                        c.execute("INSERT INTO lectures (exam_id, name, slide_count) VALUES (%s,%s,%s) RETURNING id", (e_map[sel_e], uploaded.name, 0))
                        lid = c.fetchone()[0]
                        conn.commit()
                        images = save_slides_locally(uploaded, lid)
                        c.execute("UPDATE lectures SET slide_count=%s WHERE id=%s", (len(images), lid))
                        conn.commit()
                        status.write("Generating Flashcards...")
                        for i in range(0, len(images), 10):
                            batch = images[i : i + 10]
                            new_cards, error = generate_cards_batch_json(batch, objs, i)
                            if error:
                                status.write(f"AI flashcard error: {error}")
                            if new_cards:
                                for f, b in new_cards:
                                    c.execute("INSERT INTO cards (lecture_id, front, back, next_review) VALUES (%s,%s,%s,%s)", (lid, f, b, date.today()))
                                conn.commit()
                    status.update(label="Complete!", state="complete", expanded=False)
                    st.success("Upload Finished!"); time.sleep(1); st.rerun()

    with tab_manage:
        new_class = st.text_input("Create New Class Name")
        if st.button("Create Class"):
            if not new_class.strip():
                st.warning("Class name cannot be empty.")
            else:
                c.execute("INSERT INTO classes (name) VALUES (%s)", (new_class.strip(),))
                conn.commit(); st.rerun()
        st.write("---"); st.write("🗑️ **Danger Zone**")
        c.execute("SELECT id, name FROM exams")
        all_exams = c.fetchall()
        if all_exams:
            e_del_map = {n: i for i, n in all_exams}
            del_target = st.selectbox("Select Topic to Delete", list(e_del_map.keys()))
            if st.button("Permanently Delete Topic"):
                eid = e_del_map[del_target]
                c.execute("SELECT id FROM lectures WHERE exam_id=%s", (eid,))
                l_ids = c.fetchall()
                for (lid,) in l_ids: shutil.rmtree(os.path.join(SLIDE_DIR, str(lid)), ignore_errors=True)
                c.execute("DELETE FROM exams WHERE id=%s", (eid,)); conn.commit(); st.rerun()

# --- 3. ACTIVE LEARNING ---
elif nav == "Active Learning":
    st.title("👨‍🏫 Active Learning")
    c.execute("SELECT l.id, l.name, e.name FROM lectures l JOIN exams e ON l.exam_id = e.id")
    all_lecs = c.fetchall()
    if not all_lecs: st.info("No lectures found.")
    else:
        l_ids, l_labels = [l[0] for l in all_lecs], [f"{l[1]} ({l[2]})" for l in all_lecs]
        default_idx = l_ids.index(st.session_state.active_lecture_id) if 'active_lecture_id' in st.session_state and st.session_state.active_lecture_id in l_ids else 0
        sel_label = st.selectbox("Current Lecture", l_labels, index=default_idx)
        lid = l_ids[l_labels.index(sel_label)]
        lec_path = os.path.join(SLIDE_DIR, str(lid))
        if os.path.exists(lec_path):
            slides = sorted([f for f in os.listdir(lec_path) if f.endswith(".png")], key=lambda x: int(x.split('_')[1].split('.')[0]))
            if 'read_idx' not in st.session_state: st.session_state.read_idx = 0
            start = st.session_state.read_idx
            col_slides, col_tools = st.columns([2, 1])
            with col_slides:
                end = min(start + 5, len(slides))
                st.caption(f"Slides {start+1}-{end} of {len(slides)}"); st.progress(end/len(slides))
                current_images = []
                for i in range(start, end):
                    img = PIL.Image.open(os.path.join(lec_path, slides[i]))
                    current_images.append(img); st.image(img, use_container_width=True)
                c_prev, c_next = st.columns(2)
                if c_prev.button("⬅️ Previous", use_container_width=True) and start > 0:
                    st.session_state.read_idx = max(0, start - 5); st.session_state.quiz_data = None; st.rerun()
                if c_next.button("Next ➡️", use_container_width=True) and end < len(slides):
                    st.session_state.read_idx = end; st.session_state.quiz_data = None; st.rerun()
            with col_tools:
                st.write("#### 🧠 Quick Quiz")
                if st.button("Generate Quiz", type="primary"):
                     with st.spinner("AI thinking..."): st.session_state.quiz_data = generate_interactive_quiz(current_images)
                if 'quiz_data' in st.session_state and st.session_state.quiz_data:
                    q_data = st.session_state.quiz_data
                    if "error" in q_data[0]:
                        st.error(f"AI Error: {q_data[0]['error']}")
                    else:
                        for i, q in enumerate(q_data):
                            with st.expander(f"Q{i+1}: {q['question']}", expanded=True):
                                ans = st.radio("Select:", q['options'], key=f"q_{i}")
                                if st.button("Check", key=f"chk_{i}"):
                                    corr = q['options'][q['correct_index']]
                                    if ans == corr: st.success("Correct!")
                                    else: st.error(f"Wrong. Answer: {corr}"); st.info(q['explanation'])

# --- 4. EDITOR ---
elif nav == "Editor":
    st.title("🛠️ Card Editor")
    c.execute("SELECT id, name FROM exams")
    exams = c.fetchall()
    if exams:
        e_map = {name: id for id, name in exams}
        filter_exam = st.selectbox("Topic", list(e_map.keys()))
        eid = e_map[filter_exam]
        query = "SELECT c.id, c.front, c.back FROM cards c JOIN lectures l ON c.lecture_id = l.id WHERE l.exam_id = %s"
        df = pd.read_sql(query, conn, params=(eid,))
        edited = st.data_editor(df, num_rows="dynamic", key="editor", use_container_width=True)
        if st.button("Save Changes", type="primary"):
            for i, row in edited.iterrows():
                c.execute("UPDATE cards SET front=%s, back=%s WHERE id=%s", (row['front'], row['back'], int(row['id'])))
            conn.commit(); st.toast("Saved successfully!", icon="✅")
    else: st.info("No topics found.")
