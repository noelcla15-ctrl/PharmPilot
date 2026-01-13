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
# ⚙️ CONFIGURATION
# ==========================================
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
# 🗄️ DATABASE (POSTGRESQL / SUPABASE)
# ==========================================
def get_db_connection():
    """Connects to Supabase"""
    return psycopg2.connect(DB_URL)

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
st.set_page_config(page_title="PharmPilot Cloud", page_icon="☁️", layout="wide")
st.markdown("""<style>.stButton>button {width: 100%; border-radius: 5px;}</style>""", unsafe_allow_html=True)

st.sidebar.title("☁️ PharmPilot")
st.sidebar.caption("Connected to Supabase")

if 'main_nav' not in st.session_state:
    st.session_state.main_nav = "Study Dashboard"

menu = st.sidebar.radio("Navigation", 
                        ["Study Dashboard", "📂 Library (Folders)", "👨‍🏫 Active Learning", "Card Manager"],
                        key="main_nav")

try:
    conn = get_db_connection()
    c = conn.cursor()
except Exception as e:
    st.error(f"Database Connection Failed: {e}")
    st.stop()

# CALLBACKS
def open_lecture_callback(lid):
    st.session_state.active_lecture_id = lid
    st.session_state.main_nav = "👨‍🏫 Active Learning"

# ------------------------------------------
# 1. STUDY DASHBOARD (ALGORITHM UPGRADE v4.0)
# ------------------------------------------
if menu == "Study Dashboard":
    st.header("🧠 Daily Review")
    today = date.today()
    
    # 1. PREPARE THE QUEUE
    # We prioritize cards with specific exam dates approaching
    upcoming_exams = c.execute("""
        SELECT e.id, e.exam_date, e.name 
        FROM exams e 
        WHERE e.exam_date >= %s 
        ORDER BY e.exam_date ASC
    """, (today,)).fetchall()
    
    # Check for "Crisis Mode" (Exams within 4 days)
    crisis_mode = False
    crisis_exam_ids = []
    if upcoming_exams:
        for eid, edate, ename in upcoming_exams:
            days_until = (edate - today).days
            if days_until <= 4:
                crisis_mode = True
                crisis_exam_ids.append(eid)
                st.error(f"🚨 CRISIS: {ename} is in {days_until} days!")

    # 2. SELECT CARDS
    # If crisis, pull ALL cards for that exam regardless of due date (Cram mode)
    if crisis_mode and crisis_exam_ids:
        placeholders = ",".join(["%s"] * len(crisis_exam_ids))
        QUERY = f"""
            SELECT c.id, c.front, c.back, c.interval, c.ease, c.review_count, l.exam_id, e.exam_date
            FROM cards c 
            JOIN lectures l ON c.lecture_id = l.id 
            JOIN exams e ON l.exam_id = e.id
            WHERE e.id IN ({placeholders})
            ORDER BY random() LIMIT 50
        """
        params = tuple(crisis_exam_ids)
    else:
        # Standard Mode: Only due cards
        st.success("🟢 Standard Schedule")
        QUERY = """
            SELECT c.id, c.front, c.back, c.interval, c.ease, c.review_count, l.exam_id, e.exam_date
            FROM cards c 
            JOIN lectures l ON c.lecture_id = l.id 
            JOIN exams e ON l.exam_id = e.id
            WHERE c.next_review <= %s 
            ORDER BY c.next_review ASC LIMIT 50
        """
        params = (today,)
    
    c.execute(QUERY, params)
    cards_due = c.fetchall()
    
    # 3. STUDY INTERFACE
    if not cards_due:
        st.info("🎉 No cards due! Go touch some grass.")
    else:
        if 'idx' not in st.session_state: st.session_state.idx = 0
        if 'show' not in st.session_state: st.session_state.show = False
        
        # Progress Bar
        st.progress(min((st.session_state.idx) / len(cards_due), 1.0))
        st.caption(f"Card {st.session_state.idx + 1} of {len(cards_due)}")

        if st.session_state.idx < len(cards_due):
            card = cards_due[st.session_state.idx]
            cid, front, back, interval, ease, revs, eid, exam_date = card
            
            # LEECH WARNING
            if revs > 6 and interval < 2:
                st.warning("⚠️ LEECH: You have failed this card many times. Consider rewriting it.")

            # Front Card
            st.markdown(f"""
            <div style="padding:20px;border:1px solid #ccc;border-radius:10px;background:#ffffff; color:#000000; font-size:18px;">
                <strong>Q:</strong> {front}
            </div>
            """, unsafe_allow_html=True)
            
            # Back Card (Revealed)
            if st.session_state.show:
                st.markdown(f"""
                <div style="padding:20px;margin-top:10px;border:1px solid #a8d5e2;border-radius:10px;background:#f0f8ff; color:#000000; font-size:18px;">
                    <strong>A:</strong> {back}
                </div>
                """, unsafe_allow_html=True)
                
                # --- ALGORITHM CORE ---
                def answer(quality):
                    # SM-2++ LOGIC
                    new_ease = ease
                    new_interval = interval
                    
                    if quality == 0: # FAIL
                        new_interval = 1
                        new_ease = max(1.3, ease - 0.2) # Penalty
                    
                    elif quality == 3: # HARD (Pass but difficult)
                        new_interval = max(1, int(interval * 1.2)) # Slow growth
                        new_ease = max(1.3, ease - 0.15) # Slight penalty
                        
                    elif quality == 4: # GOOD (Standard)
                        new_interval = max(1, int(interval * ease))
                        # Ease stays same
                        
                    elif quality == 5: # EASY (Bonus)
                        new_interval = max(1, int(interval * ease * 1.3)) # Bonus multiplier
                        new_ease = min(3.0, ease + 0.15) # Boost ease

                    # --- EXAM CAP LOGIC (The Pharmacy Special) ---
                    if exam_date:
                        days_until_exam = (exam_date - today).days
                        # If the new interval pushes past the exam, cap it
                        if days_until_exam > 0 and new_interval >= days_until_exam:
                            new_interval = max(1, days_until_exam - 1) # Show day before exam
                    
                    # UPDATE DB
                    c.execute("""
                        UPDATE cards 
                        SET next_review=%s, interval=%s, ease=%s, review_count=%s 
                        WHERE id=%s
                    """, (today + timedelta(days=new_interval), new_interval, new_ease, revs + 1, cid))
                    conn.commit()
                    
                    # Next Card
                    st.session_state.show = False
                    st.session_state.idx += 1
                    st.rerun()

                # Buttons
                c1, c2, c3, c4 = st.columns(4)
                if c1.button("❌ Again (1d)", use_container_width=True): answer(0)
                if c2.button("😓 Hard", use_container_width=True): answer(3)
                if c3.button("✅ Good", use_container_width=True): answer(4)
                if c4.button("🚀 Easy", use_container_width=True): answer(5)
            
            else:
                if st.button("Show Answer", type="primary", use_container_width=True): 
                    st.session_state.show = True
                    st.rerun()
        else:
            st.balloons()
            st.success("🎉 You are done for the day!")
            if st.button("Start Over"):
                st.session_state.idx = 0
                st.rerun()
                
# ------------------------------------------
# 2. LIBRARY (FOLDERS)
# ------------------------------------------
elif menu == "📂 Library (Folders)":
    today = date.today()
    col_title, col_up = st.columns([3, 1])
    with col_title: st.header("📂 My Library")
    with col_up: show_upload = st.checkbox("➕ Upload New")

    if show_upload:
        with st.container():
            st.markdown("### 📤 Upload Center")
            
            with st.expander("⚙️ Manage Topics"):
                c.execute("SELECT id, name FROM classes")
                m_classes = c.fetchall()
                if m_classes:
                    m_c_map = {n: i for i, n in m_classes}
                    m_sel_c = st.selectbox("Class", list(m_c_map.keys()), key="man_class")
                    
                    c.execute("SELECT id, name FROM exams WHERE class_id=%s", (m_c_map[m_sel_c],))
                    m_exams = c.fetchall()
                    
                    if m_exams:
                        m_e_map = {n: i for i, n in m_exams}
                        m_sel_e = st.selectbox("Topic", list(m_e_map.keys()), key="man_exam")
                        eid = m_e_map[m_sel_e]
                        
                        c1, c2 = st.columns(2)
                        if c1.button("🗑️ Delete Topic"):
                            c.execute("SELECT id FROM lectures WHERE exam_id=%s", (eid,))
                            l_ids = c.fetchall()
                            for (lid,) in l_ids:
                                shutil.rmtree(os.path.join(SLIDE_DIR, str(lid)), ignore_errors=True)
                            
                            c.execute("DELETE FROM exams WHERE id=%s", (eid,))
                            conn.commit(); st.rerun()
                        
                        new_n = c2.text_input("Rename", value=m_sel_e)
                        if c2.button("Update"):
                            c.execute("UPDATE exams SET name=%s WHERE id=%s", (new_n, eid))
                            conn.commit(); st.rerun()

            st.markdown("---")

            c.execute("SELECT id, name FROM classes")
            classes = c.fetchall()
            with st.expander("Create New Class"):
                new_c = st.text_input("Class Name")
                if st.button("Save Class"):
                    c.execute("INSERT INTO classes (name) VALUES (%s)", (new_c,))
                    conn.commit(); st.rerun()
            
            if classes:
                c_map = {n: i for i, n in classes}
                sel_c = st.selectbox("Select Class", list(c_map.keys()))
                col1, col2 = st.columns(2)
                with col1: new_topic = st.text_input("New Topic/Exam Name")
                with col2: 
                    tbd = st.checkbox("Date TBD", value=True)
                    d_val = None if tbd else st.date_input("Date")
                if st.button("Add Topic"):
                    c.execute("INSERT INTO exams (class_id, name, exam_date) VALUES (%s,%s,%s)", 
                              (c_map[sel_c], new_topic, d_val))
                    conn.commit(); st.rerun()
                
                c.execute("SELECT id, name FROM exams WHERE class_id=%s", (c_map[sel_c],))
                exams = c.fetchall()
                if exams:
                    e_map = {n: i for i, n in exams}
                    sel_e = st.selectbox("Select Topic", list(e_map.keys()))
                    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
                    objs = st.text_area("Learning Objectives")
                    
                    if uploaded_files and st.button("🚀 Process Files"):
                        status = st.empty()
                        main_bar = st.progress(0)
                        total_files = len(uploaded_files)
                        
                        for f_idx, uploaded in enumerate(uploaded_files):
                            status.info(f"Processing {uploaded.name}...")
                            
                            c.execute("INSERT INTO lectures (exam_id, name, slide_count) VALUES (%s,%s,%s) RETURNING id", 
                                      (e_map[sel_e], uploaded.name, 0))
                            lid = c.fetchone()[0]
                            conn.commit()
                            
                            images = save_slides_locally(uploaded, lid)
                            c.execute("UPDATE lectures SET slide_count=%s WHERE id=%s", (len(images), lid))
                            conn.commit()
                            
                            log = st.container()
                            BATCH_SIZE = 10
                            for i in range(0, len(images), BATCH_SIZE):
                                batch = images[i : i + BATCH_SIZE]
                                new_cards, error_msg = generate_cards_batch_json(batch, objs, i)
                                
                                if new_cards:
                                    for f, b in new_cards:
                                        c.execute("INSERT INTO cards (lecture_id, front, back, next_review) VALUES (%s,%s,%s,%s)", 
                                                  (lid, f, b, today))
                                    conn.commit()
                                    with log:
                                        st.caption(f"✅ Batch {i//BATCH_SIZE + 1}: +{len(new_cards)} cards")
                                else:
                                    if error_msg:
                                        with log:
                                            st.caption(f"⚠️ Batch Error: {error_msg}")
                                    else:
                                        with log:
                                            st.caption(f"⏺️ Batch {i//BATCH_SIZE + 1}: No cards")
                                
                            main_bar.progress((f_idx + 1) / total_files)
                        status.success("All files processed!")

    st.write("Browse your materials:")
    c.execute("SELECT id, name FROM classes")
    classes = c.fetchall()
    for class_id, class_name in classes:
        with st.expander(f"📁 {class_name}", expanded=False):
            c.execute("SELECT id, name FROM exams WHERE class_id=%s", (class_id,))
            exams = c.fetchall()
            if not exams: st.caption("No topics yet.")
            for exam_id, exam_name in exams:
                st.markdown(f"**&nbsp;&nbsp;&nbsp;📂 {exam_name}**")
                c.execute("SELECT id, name, slide_count FROM lectures WHERE exam_id=%s", (exam_id,))
                lectures = c.fetchall()
                if not lectures: st.caption("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (No lectures)")
                
                # REGENERATION OBJECTIVES (Generic)
                default_objs = "Exam relevant facts. MOA, Dosing, Interactions."
                
                for lid, lname, lcount in lectures:
                    # THREE COLUMNS: INFO | OPEN | RETRY
                    col_txt, col_open, col_retry = st.columns([0.6, 0.2, 0.2])
                    
                    with col_txt:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;📄 {lname} *({lcount} slides)*")
                    
                    with col_open:
                        st.button("📖 Open", 
                                 key=f"open_{lid}", 
                                 on_click=open_lecture_callback, 
                                 args=(lid,))
                    
                    with col_retry:
                        # NEW: Retry Button
                        if st.button("⚡ AI", key=f"retry_{lid}", help="Regenerate cards"):
                            lec_path = os.path.join(SLIDE_DIR, str(lid))
                            if os.path.exists(lec_path):
                                slides = sorted([f for f in os.listdir(lec_path) if f.endswith(".png")], 
                                                key=lambda x: int(x.split('_')[1].split('.')[0]))
                                
                                images = [PIL.Image.open(os.path.join(lec_path, s)) for s in slides]
                                
                                st.toast(f"Regenerating {len(images)} slides...", icon="⚡")
                                log = st.container()
                                progress = st.progress(0)
                                BATCH_SIZE = 10
                                
                                for i in range(0, len(images), BATCH_SIZE):
                                    batch = images[i : i + BATCH_SIZE]
                                    new_cards, error_msg = generate_cards_batch_json(batch, default_objs, i)
                                    
                                    if new_cards:
                                        for f, b in new_cards:
                                            c.execute("INSERT INTO cards (lecture_id, front, back, next_review) VALUES (%s,%s,%s,%s)", 
                                                      (lid, f, b, today))
                                        conn.commit()
                                        with log: 
                                            st.caption(f"✅ Batch {i//BATCH_SIZE + 1}: +{len(new_cards)} cards")
                                    else:
                                        if error_msg:
                                            with log:
                                                st.caption(f"⚠️ Batch Error: {error_msg}")
                                        else:
                                            with log:
                                                st.caption(f"⏺️ Batch {i//BATCH_SIZE + 1}: No cards")
                                    
                                    progress.progress(min((i + BATCH_SIZE) / len(images), 1.0))
                                
                                st.success("Done! Refreshing...")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("No saved slides found. Delete and re-upload.")

# ------------------------------------------
# 3. ACTIVE LEARNING
# ------------------------------------------
elif menu == "👨‍🏫 Active Learning":
    st.header("👨‍🏫 Scroll & Quiz Mode")
    
    c.execute("SELECT l.id, l.name, e.name FROM lectures l JOIN exams e ON l.exam_id = e.id")
    all_lecs = c.fetchall()
    if not all_lecs:
        st.warning("No lectures found.")
    else:
        lecture_ids = [l[0] for l in all_lecs]
        lecture_labels = [f"{l[1]} ({l[2]})" for l in all_lecs]
        
        default_index = 0
        if 'active_lecture_id' in st.session_state and st.session_state.active_lecture_id in lecture_ids:
            default_index = lecture_ids.index(st.session_state.active_lecture_id)
        
        sel_label = st.selectbox("Choose Lecture", lecture_labels, index=default_index)
        lid = lecture_ids[lecture_labels.index(sel_label)]
        
        lec_path = os.path.join(SLIDE_DIR, str(lid))
        
        if os.path.exists(lec_path):
            slides = sorted([f for f in os.listdir(lec_path) if f.endswith(".png")], 
                            key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            col_set1, col_set2 = st.columns(2)
            with col_set1: 
                chunk_size = st.number_input("Slides to read at once", min_value=1, value=5)
            
            if 'read_idx' not in st.session_state: st.session_state.read_idx = 0
            if st.session_state.read_idx >= len(slides): st.session_state.read_idx = 0
                
            start = st.session_state.read_idx
            end = min(start + chunk_size, len(slides))
            
            st.markdown(f"### 📖 Reading Slides {start+1} - {end} (of {len(slides)})")
            st.progress(end / len(slides))
            
            current_images = []
            for i in range(start, end):
                img_path = os.path.join(lec_path, slides[i])
                img = PIL.Image.open(img_path)
                current_images.append(img)
                st.image(img, use_container_width=True)
                st.caption(f"Slide {i+1}")
                st.markdown("---")
            
            if 'quiz_data' not in st.session_state: st.session_state.quiz_data = None
            
            c_prev, c_quiz, c_next = st.columns([1, 2, 1])
            
            with c_prev:
                if start > 0:
                    if st.button("⬅️ Previous"):
                        st.session_state.read_idx = max(0, start - chunk_size)
                        st.session_state.quiz_data = None
                        st.rerun()
            
            with c_quiz:
                if st.button("🧠 Quiz Me on This Chunk", use_container_width=True):
                    with st.spinner("Generating Quiz..."):
                        st.session_state.quiz_data = generate_interactive_quiz(current_images)
            
            with c_next:
                if end < len(slides):
                    if st.button("Next Chunk ➡️"):
                        st.session_state.read_idx = end
                        st.session_state.quiz_data = None
                        st.rerun()
                else:
                    st.success("Finished!")

            if st.session_state.quiz_data:
                st.markdown("### 📝 Interactive Quiz")
                if "error" in st.session_state.quiz_data[0]:
                    st.error(f"Error: {st.session_state.quiz_data[0]['error']}")
                else:
                    for i, q_item in enumerate(st.session_state.quiz_data):
                        st.markdown(f"#### {i+1}. {q_item['question']}")
                        user_choice = st.radio("Answer:", q_item['options'], key=f"q_{i}", index=None)
                        if user_choice:
                            if st.button(f"Check Answer {i+1}", key=f"btn_{i}"):
                                correct_idx = q_item['correct_index']
                                correct_str = q_item['options'][correct_idx]
                                if user_choice == correct_str: st.success("✅ Correct!")
                                else: st.error(f"❌ Incorrect. Answer: {correct_str}")
                                st.info(f"**Explanation:** {q_item['explanation']}")
                        st.markdown("---")

# ------------------------------------------
# 4. CARD MANAGER
# ------------------------------------------
elif menu == "Card Manager":
    st.header("🛠️ Card Editor")
    
    c.execute("SELECT id, name FROM exams")
    exams = c.fetchall()
    if exams:
        e_map = {name: id for id, name in exams}
        filter_exam = st.selectbox("Filter by Topic", ["All"] + list(e_map.keys()))
        
        if filter_exam == "All":
            query = "SELECT id, front, back FROM cards"
            params = ()
        else:
            eid = e_map[filter_exam]
            query = """
                SELECT c.id, c.front, c.back 
                FROM cards c 
                JOIN lectures l ON c.lecture_id = l.id 
                WHERE l.exam_id = %s
            """
            params = (eid,)
            
        df = pd.read_sql(query, conn, params=params)
        edited = st.data_editor(df, num_rows="dynamic", key="editor")
        
        if st.button("Save Changes"):
            for i, row in edited.iterrows():
                c.execute("UPDATE cards SET front=%s, back=%s WHERE id=%s", (row['front'], row['back'], int(row['id'])))
            conn.commit(); st.success("Saved!")
    else:
        st.info("No cards found.")

conn.close()
