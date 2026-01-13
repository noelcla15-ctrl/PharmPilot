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
# ⚙️ CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="PharmPilot", page_icon="💊", layout="centered")

# 🎨 CUSTOM UI STYLING
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Flashcard Style */
    .flashcard {
        background-color: white;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 6px solid #4F8BF9;
        font-size: 20px;
        margin-bottom: 20px;
        color: #1f1f1f;
    }
    
    .flashcard-back {
        background-color: #eef6ff;
        border-left: 6px solid #00c853;
        color: #1f1f1f;
    }
    
    /* Button Styling */
    .stButton>button {
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        border: none;
        transition: transform 0.1s;
    }
    .stButton>button:active {
        transform: scale(0.98);
    }
    
    /* Clean Expanders */
    div[data-testid="stExpander"] {
        border: none;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.08);
        background: white;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    DB_URL = st.secrets["SUPABASE_DB_URL"]
except Exception:
    st.error("Missing Secrets! Make sure .streamlit/secrets.toml exists.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# SAFETY SETTINGS
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- ENGINES ---
flash_model = genai.GenerativeModel('gemini-2.5-flash', safety_settings=safety_settings)
quiz_model = genai.GenerativeModel('gemini-2.5-pro', safety_settings=safety_settings)

SLIDE_DIR = "lecture_slides"
if not os.path.exists(SLIDE_DIR):
    os.makedirs(SLIDE_DIR)

# ==========================================
# 🗄️ DATABASE
# ==========================================
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
    Structure:
    [
        {{"front": "Question...", "back": "Answer..."}}
    ]
    """
    
    try:
        response = flash_model.generate_content([prompt] + images)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        
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
    Structure: 
    [
        {
            "question": "...",
            "options": ["A) ...", "B) ..."],
            "correct_index": 0,
            "explanation": "..."
        }
    ]
    """
    try:
        content = [prompt] + images
        response = quiz_model.generate_content(content)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        return [{"error": str(e)}]

# ==========================================
# 🖥️ UI LAYOUT
# ==========================================
st.sidebar.title("💊 PharmPilot")
st.sidebar.markdown("---")

if 'main_nav' not in st.session_state:
    st.session_state.main_nav = "Review"

# Clean Sidebar Menu
nav = st.sidebar.radio("Menu", 
    ["Review", "Library", "Active Learning", "Editor"],
    key="main_nav"
)

# Callbacks
def open_lecture_callback(lid):
    st.session_state.active_lecture_id = lid
    st.session_state.main_nav = "Active Learning"

# ------------------------------------------
# 1. REVIEW DASHBOARD (Optimized UI)
# ------------------------------------------
if nav == "Review":
    st.title("🧠 Daily Review")
    
    today = date.today()
    c.execute("SELECT id FROM exams WHERE exam_date >= %s ORDER BY exam_date ASC LIMIT 1", (today,))
    next_exam = c.fetchone()
    
    # Logic to fetch cards
    c.execute("""
        SELECT c.id, c.front, c.back, c.interval, c.ease, c.review_count, l.exam_id, e.exam_date
        FROM cards c 
        JOIN lectures l ON c.lecture_id = l.id 
        JOIN exams e ON l.exam_id = e.id
        WHERE c.next_review <= %s 
        ORDER BY c.next_review ASC LIMIT 50
    """, (today,))
    cards_due = c.fetchall()
    
    if not cards_due:
        st.canvas = st.empty()
        st.balloons()
        st.success("🎉 You are all caught up!")
        st.info("Check the Library to add more lectures.")
    else:
        if 'idx' not in st.session_state: st.session_state.idx = 0
        if 'show' not in st.session_state: st.session_state.show = False
        
        # Progress Bar
        progress = min((st.session_state.idx) / len(cards_due), 1.0)
        st.progress(progress)
        
        if st.session_state.idx < len(cards_due):
            card = cards_due[st.session_state.idx]
            cid, front, back, interval, ease, revs, eid, exam_date = card
            
            # LEECH WARNING
            if revs > 6 and interval < 2:
                st.warning("⚠️ Difficult Card (Leech)")

            # FRONT CARD
            st.markdown(f"""
            <div class="flashcard">
                <div style="font-size:14px; color:#888; margin-bottom:10px;">QUESTION</div>
                {front}
            </div>
            """, unsafe_allow_html=True)
            
            # BACK CARD
            if st.session_state.show:
                st.markdown(f"""
                <div class="flashcard flashcard-back">
                     <div style="font-size:14px; color:#888; margin-bottom:10px;">ANSWER</div>
                    {back}
                </div>
                """, unsafe_allow_html=True)
                
                # ALGORITHM
                def answer(quality):
                    new_ease = ease
                    new_interval = interval
                    
                    if quality == 0: # FAIL
                        new_interval = 1
                        new_ease = max(1.3, ease - 0.2)
                    elif quality == 3: # HARD
                        new_interval = max(1, int(interval * 1.2))
                        new_ease = max(1.3, ease - 0.15)
                    elif quality == 4: # GOOD
                        new_interval = max(1, int(interval * ease))
                    elif quality == 5: # EASY
                        new_interval = max(1, int(interval * ease * 1.3))
                        new_ease = min(3.0, ease + 0.15)

                    if exam_date:
                        days_until_exam = (exam_date - today).days
                        if days_until_exam > 0 and new_interval >= days_until_exam:
                            new_interval = max(1, days_until_exam - 1)
                    
                    c.execute("""UPDATE cards SET next_review=%s, interval=%s, ease=%s, review_count=%s WHERE id=%s""", 
                             (today + timedelta(days=new_interval), new_interval, new_ease, revs + 1, cid))
                    conn.commit()
                    st.session_state.show = False
                    st.session_state.idx += 1
                    st.rerun()

                # ACTION BUTTONS
                c1, c2, c3, c4 = st.columns(4)
                if c1.button("❌ Fail", use_container_width=True): answer(0)
                if c2.button("😓 Hard", use_container_width=True): answer(3)
                if c3.button("✅ Good", use_container_width=True): answer(4)
                if c4.button("🚀 Easy", use_container_width=True): answer(5)

            else:
                if st.button("Show Answer", type="primary", use_container_width=True): 
                    st.session_state.show = True
                    st.rerun()
        else:
            st.success("Session Complete!")
            if st.button("Start Over"):
                st.session_state.idx = 0
                st.rerun()

# ------------------------------------------
# 2. LIBRARY (Tabs UI)
# ------------------------------------------
elif nav == "Library":
    st.title("📂 Library")
    
    tab_browse, tab_upload, tab_manage = st.tabs(["📚 Browse Materials", "☁️ Upload New", "⚙️ Manage"])
    
    # TAB 1: BROWSE
    with tab_browse:
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
            st.info("Library is empty. Go to 'Upload New' to start!")
        else:
            for class_name, class_group in df.groupby("class_name"):
                with st.expander(f"📁 {class_name}", expanded=False):
                    for exam_name, exam_group in class_group.groupby("exam_name"):
                        st.caption(f"📂 {exam_name}")
                        for _, row in exam_group.iterrows():
                            if pd.notna(row['lecture_name']):
                                lid = int(row['lecture_id'])
                                lname = row['lecture_name']
                                lcount = int(row['slide_count'])
                                
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
                                                new_cards, _ = generate_cards_batch_json(batch, "Review", i)
                                                if new_cards:
                                                    for f, b in new_cards:
                                                        c.execute("INSERT INTO cards (lecture_id, front, back, next_review) VALUES (%s,%s,%s,%s)", (lid, f, b, date.today()))
                                                    conn.commit()
                                            st.rerun()
                        st.divider()

    # TAB 2: UPLOAD
    with tab_upload:
        c.execute("SELECT id, name FROM classes")
        classes = c.fetchall()
        
        if not classes:
            st.warning("Create a Class in 'Manage' tab first!")
        else:
            c_map = {n: i for i, n in classes}
            sel_c = st.selectbox("Select Class", list(c_map.keys()))
            
            c.execute("SELECT id, name FROM exams WHERE class_id=%s", (c_map[sel_c],))
            exams = c.fetchall()
            
            # QUICK ADD EXAM
            with st.expander("➕ Add New Topic to this Class"):
                new_topic = st.text_input("Topic Name")
                d_val = st.date_input("Exam Date")
                if st.button("Create Topic"):
                    c.execute("INSERT INTO exams (class_id, name, exam_date) VALUES (%s,%s,%s)", 
                              (c_map[sel_c], new_topic, d_val))
                    conn.commit(); st.rerun()

            if exams:
                e_map = {n: i for i, n in exams}
                sel_e = st.selectbox("Select Topic to Upload PDFs", list(e_map.keys()))
                uploaded_files = st.file_uploader("Drop PDFs Here", type="pdf", accept_multiple_files=True)
                objs = st.text_area("Learning Objectives (Optional)")
                
                if uploaded_files and st.button("🚀 Upload & Process", type="primary"):
                    status = st.status("Processing...", expanded=True)
                    total = len(uploaded_files)
                    for idx, uploaded in enumerate(uploaded_files):
                        status.write(f"Reading {uploaded.name}...")
                        c.execute("INSERT INTO lectures (exam_id, name, slide_count) VALUES (%s,%s,%s) RETURNING id", 
                                  (e_map[sel_e], uploaded.name, 0))
                        lid = c.fetchone()[0]
                        conn.commit()
                        images = save_slides_locally(uploaded, lid)
                        c.execute("UPDATE lectures SET slide_count=%s WHERE id=%s", (len(images), lid))
                        conn.commit()
                        
                        status.write("Generating Flashcards...")
                        for i in range(0, len(images), 10):
                            batch = images[i : i + 10]
                            new_cards, _ = generate_cards_batch_json(batch, objs, i)
                            if new_cards:
                                for f, b in new_cards:
                                    c.execute("INSERT INTO cards (lecture_id, front, back, next_review) VALUES (%s,%s,%s,%s)", 
                                              (lid, f, b, date.today()))
                                conn.commit()
                    status.update(label="Complete!", state="complete", expanded=False)
                    st.success("Upload Finished!")
                    time.sleep(1)
                    st.rerun()

    # TAB 3: MANAGE
    with tab_manage:
        new_class = st.text_input("Create New Class Name")
        if st.button("Create Class"):
            c.execute("INSERT INTO classes (name) VALUES (%s)", (new_class,))
            conn.commit(); st.rerun()
        
        st.write("---")
        st.write("🗑️ **Danger Zone**")
        c.execute("SELECT id, name FROM exams")
        all_exams = c.fetchall()
        if all_exams:
            e_del_map = {n: i for i, n in all_exams}
            del_target = st.selectbox("Select Topic to Delete", list(e_del_map.keys()))
            if st.button("Permanently Delete Topic"):
                eid = e_del_map[del_target]
                c.execute("SELECT id FROM lectures WHERE exam_id=%s", (eid,))
                l_ids = c.fetchall()
                for (lid,) in l_ids:
                    shutil.rmtree(os.path.join(SLIDE_DIR, str(lid)), ignore_errors=True)
                c.execute("DELETE FROM exams WHERE id=%s", (eid,))
                conn.commit(); st.rerun()

# ------------------------------------------
# 3. ACTIVE LEARNING
# ------------------------------------------
elif nav == "Active Learning":
    st.title("👨‍🏫 Active Learning")
    
    c.execute("SELECT l.id, l.name, e.name FROM lectures l JOIN exams e ON l.exam_id = e.id")
    all_lecs = c.fetchall()
    
    if not all_lecs:
        st.info("No lectures found.")
    else:
        l_ids = [l[0] for l in all_lecs]
        l_labels = [f"{l[1]} ({l[2]})" for l in all_lecs]
        
        default_idx = 0
        if 'active_lecture_id' in st.session_state and st.session_state.active_lecture_id in l_ids:
            default_idx = l_ids.index(st.session_state.active_lecture_id)
            
        sel_label = st.selectbox("Current Lecture", l_labels, index=default_idx)
        lid = l_ids[l_labels.index(sel_label)]
        
        lec_path = os.path.join(SLIDE_DIR, str(lid))
        if os.path.exists(lec_path):
            slides = sorted([f for f in os.listdir(lec_path) if f.endswith(".png")], 
                            key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            if 'read_idx' not in st.session_state: st.session_state.read_idx = 0
            start = st.session_state.read_idx
            
            # UI: SPLIT VIEW
            col_slides, col_tools = st.columns([2, 1])
            
            with col_slides:
                end = min(start + 5, len(slides))
                st.caption(f"Slides {start+1}-{end} of {len(slides)}")
                st.progress(end/len(slides))
                
                # Fetch images for quiz generation
                current_images = []
                for i in range(start, end):
                    img_path = os.path.join(lec_path, slides[i])
                    img = PIL.Image.open(img_path)
                    current_images.append(img)
                    st.image(img, use_container_width=True)
                
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
                if st.button("Generate Quiz for these slides", type="primary"):
                     with st.spinner("AI thinking..."):
                        st.session_state.quiz_data = generate_interactive_quiz(current_images)
                
                if 'quiz_data' in st.session_state and st.session_state.quiz_data:
                    q_data = st.session_state.quiz_data
                    if "error" in q_data[0]:
                        st.error("AI Error")
                    else:
                        for i, q in enumerate(q_data):
                            with st.expander(f"Q{i+1}: {q['question']}", expanded=True):
                                ans = st.radio("Select:", q['options'], key=f"q_{i}")
                                if st.button("Check", key=f"chk_{i}"):
                                    corr = q['options'][q['correct_index']]
                                    if ans == corr: st.success("Correct!")
                                    else: st.error(f"Wrong. Answer: {corr}")
                                    st.info(q['explanation'])

# ------------------------------------------
# 4. EDITOR
# ------------------------------------------
elif nav == "Editor":
    st.title("🛠️ Card Editor")
    
    c.execute("SELECT id, name FROM exams")
    exams = c.fetchall()
    if exams:
        e_map = {name: id for id, name in exams}
        filter_exam = st.selectbox("Topic", list(e_map.keys()))
        eid = e_map[filter_exam]
        
        query = """
            SELECT c.id, c.front, c.back 
            FROM cards c 
            JOIN lectures l ON c.lecture_id = l.id 
            WHERE l.exam_id = %s
        """
        df = pd.read_sql(query, conn, params=(eid,))
        edited = st.data_editor(df, num_rows="dynamic", key="editor", use_container_width=True)
        
        if st.button("Save Changes", type="primary"):
            for i, row in edited.iterrows():
                c.execute("UPDATE cards SET front=%s, back=%s WHERE id=%s", (row['front'], row['back'], int(row['id'])))
            conn.commit(); st.toast("Saved successfully!", icon="✅")
    else:
        st.info("No topics found.")
