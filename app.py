import streamlit as st
import sqlite3
import google.generativeai as genai
from datetime import date, datetime, timedelta
import fitz  # PyMuPDF
import PIL.Image
import io
import time
import pandas as pd

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# ⚠️ PASTE YOUR KEY HERE
GOOGLE_API_KEY = "AIzaSyBVCvK_gPNtQMIK_W5WA2X0gr8tncJ6k-g"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==========================================
# 🗄️ DATABASE ENGINE
# ==========================================
def init_db():
    conn = sqlite3.connect('pharmpilot.db', check_same_thread=False)
    c = conn.cursor()
    
    # Core Hierarchy
    c.execute('''CREATE TABLE IF NOT EXISTS classes (id INTEGER PRIMARY KEY, name TEXT)''')
    # Updated: exam_date is now allowed to be NULL (None)
    c.execute('''CREATE TABLE IF NOT EXISTS exams 
                 (id INTEGER PRIMARY KEY, class_id INTEGER, name TEXT, exam_date DATE,
                  FOREIGN KEY(class_id) REFERENCES classes(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS lectures 
                 (id INTEGER PRIMARY KEY, exam_id INTEGER, name TEXT, raw_text TEXT,
                  FOREIGN KEY(exam_id) REFERENCES exams(id))''')
    
    # Flashcards
    c.execute('''CREATE TABLE IF NOT EXISTS cards
                 (id INTEGER PRIMARY KEY, lecture_id INTEGER, 
                  front TEXT, back TEXT, 
                  next_review DATE, interval INTEGER DEFAULT 1, ease REAL DEFAULT 2.5,
                  review_count INTEGER DEFAULT 0,
                  FOREIGN KEY(lecture_id) REFERENCES lectures(id))''')
    conn.commit()
    conn.close()

def get_db_connection():
    return sqlite3.connect('pharmpilot.db', check_same_thread=False)

# Initialize DB on first load
init_db()

# ==========================================
# 🧠 AI & LOGIC FUNCTIONS
# ==========================================
def process_file_visually(uploaded_file):
    """Converts PDF slides into Images for AI Vision"""
    images = []
    if uploaded_file.type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img = PIL.Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
    return images

def generate_cards_vision(image, objectives):
    """Sends slide image to Gemini to extract drugs/concepts"""
    prompt = f"""
    You are a Pharmacy Professor. Analyze this slide image.
    CONTEXT: Learning Objectives: {objectives}
    
    TASK:
    1. Extract high-yield concepts (Text, Tables, Structures).
    2. If a drug is listed, create cards for MOA, Indication, ADE, or Pearls.
    3. If a graph/chart is shown, interpret the trend.
    4. FORMAT: Return strict "Question|Answer" pairs (one per line).
    """
    try:
        response = model.generate_content([prompt, image])
        cards = []
        for line in response.text.split('\n'):
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    cards.append((parts[0].strip(), parts[1].strip()))
        return cards
    except:
        return []

def update_srs(card_id, quality, interval, ease, reviews):
    """SuperMemo-2 Spaced Repetition Algorithm"""
    if quality < 3: # Fail
        new_int = 1
        new_ease = ease
    else: # Pass
        new_int = int(interval * ease)
        new_ease = ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        if new_ease < 1.3: new_ease = 1.3
        
    new_date = date.today() + timedelta(days=new_int)
    conn = get_db_connection()
    conn.execute("UPDATE cards SET next_review=?, interval=?, ease=?, review_count=? WHERE id=?",
                 (new_date, new_int, new_ease, reviews + 1, card_id))
    conn.commit()
    conn.close()

# ==========================================
# 🖥️ USER INTERFACE
# ==========================================
st.set_page_config(page_title="PharmPilot Pro", page_icon="💊", layout="wide")

# Sidebar
st.sidebar.title("💊 PharmPilot")
menu = st.sidebar.radio("Go to:", ["Study Dashboard", "Add Material", "Card Manager & Editor", "Backup"])

conn = get_db_connection()
c = conn.cursor()

# ------------------------------------------
# 1. STUDY DASHBOARD (The "Do" Mode)
# ------------------------------------------
if menu == "Study Dashboard":
    st.header("🧠 Daily Review Session")
    
    # Priority Logic
    today = date.today()
    crisis_end = today + timedelta(days=4)
    
    # Check Exams (Only checks exams that actually HAVE a date)
    exams = c.execute("SELECT name, exam_date FROM exams WHERE exam_date IS NOT NULL AND exam_date BETWEEN ? AND ?", 
                      (today, crisis_end)).fetchall()
    
    if exams:
        st.error(f"🚨 CRISIS MODE: {len(exams)} Exams approaching! Prioritizing relevant cards.")
        query = """SELECT c.id, c.front, c.back, c.interval, c.ease, c.review_count 
                   FROM cards c JOIN lectures l ON c.lecture_id = l.id 
                   JOIN exams e ON l.exam_id = e.id 
                   WHERE e.exam_date BETWEEN ? AND ? ORDER BY random() LIMIT 50"""
        params = (today, crisis_end)
    else:
        st.success("🟢 Standard Schedule")
        query = "SELECT id, front, back, interval, ease, review_count FROM cards WHERE next_review <= ? LIMIT 50"
        params = (today,)
        
    due_cards = c.execute(query, params).fetchall()
    
    if not due_cards:
        st.info("🎉 You are all caught up! No cards due.")
    else:
        # Session State for Card Navigation
        if 'idx' not in st.session_state: st.session_state.idx = 0
        if 'show' not in st.session_state: st.session_state.show = False
        
        if st.session_state.idx < len(due_cards):
            card = due_cards[st.session_state.idx]
            cid, front, back, interval, ease, revs = card
            
            # Progress
            st.progress((st.session_state.idx) / len(due_cards))
            st.caption(f"Card {st.session_state.idx + 1} of {len(due_cards)}")
            
            # Front (FIXED VISIBILITY: Added color: #333333)
            st.markdown(f"""<div style="padding:20px;border:1px solid #ccc;border-radius:10px;background:#f9f9f9; color: #333333;">
                        <h3 style="margin:0;">{front}</h3></div>""", unsafe_allow_html=True)
            
            # Back & Buttons
            if st.session_state.show:
                st.markdown(f"""<div style="padding:20px;margin-top:10px;border:1px solid #a8d5e2;border-radius:10px;background:#e3f2fd; color: #333333;">
                            <h4 style="margin:0;">{back}</h4></div>""", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                if c1.button("❌ Again (Fail)"):
                    update_srs(cid, 0, interval, ease, revs)
                    st.session_state.show = False
                    st.session_state.idx += 1
                    st.rerun()
                if c2.button("😐 Hard (Pass)"):
                    update_srs(cid, 3, interval, ease, revs)
                    st.session_state.show = False
                    st.session_state.idx += 1
                    st.rerun()
                if c3.button("✅ Easy (Perfect)"):
                    update_srs(cid, 5, interval, ease, revs)
                    st.session_state.show = False
                    st.session_state.idx += 1
                    st.rerun()
            else:
                if st.button("Show Answer"):
                    st.session_state.show = True
                    st.rerun()
        else:
            st.success("Session Complete!")
            if st.button("Start Over"):
                st.session_state.idx = 0
                st.rerun()

# ------------------------------------------
# 2. ADD MATERIAL (AI & Manual)
# ------------------------------------------
elif menu == "Add Material":
    st.header("📚 Add Curriculum")
    
    tab1, tab2 = st.tabs(["🤖 AI Upload (Slides)", "✍️ Manual Entry"])
    
    # SETUP HELPERS
    classes = c.execute("SELECT id, name FROM classes").fetchall()
    
    with tab1:
        st.subheader("Upload Slides")
        # Quick Class/Exam Creator
        with st.expander("Create New Class / Exam Topic", expanded=True):
            new_class = st.text_input("New Class Name (e.g. Therapeutics)")
            if st.button("Save Class"):
                c.execute("INSERT INTO classes (name) VALUES (?)", (new_class,))
                conn.commit()
                st.success(f"Added {new_class}")
                st.rerun()

        if classes:
            c_map = {n: i for i, n in classes}
            sel_c = st.selectbox("Select Class", list(c_map.keys()), key="ai_c")
            
            exams = c.execute("SELECT id, name FROM exams WHERE class_id=?", (c_map[sel_c],)).fetchall()
            
            # --- UPDATED EXAM CREATION LOGIC ---
            if not exams:
                st.warning(f"No topics/exams found for {sel_c}.")
            
            # Always show the "Add Topic/Exam" form at the bottom
            st.markdown("---")
            st.write(f"**Add a new Topic or Exam for {sel_c}:**")
            col_e1, col_e2, col_e3 = st.columns([2, 1, 1])
            with col_e1:
                en = st.text_input("Topic/Exam Name (e.g. Cardiology)")
            with col_e2:
                # OPTIONAL DATE LOGIC
                tbd = st.checkbox("Date TBD / General Topic", value=False)
                if tbd:
                    ed = None
                else:
                    ed = st.date_input("Exam Date")
            with col_e3:
                st.write("") # Spacer
                st.write("") # Spacer
                if st.button("Add Topic"):
                    c.execute("INSERT INTO exams (class_id, name, exam_date) VALUES (?,?,?)", 
                              (c_map[sel_c], en, ed))
                    conn.commit()
                    st.rerun()
            st.markdown("---")

            # UPLOAD LOGIC
            if exams:
                e_map = {n: i for i, n in exams}
                sel_e = st.selectbox("Assign to Topic/Exam", list(e_map.keys()), key="ai_e")
                
                uploaded = st.file_uploader("Upload PDF Lecture", type="pdf")
                objs = st.text_area("Learning Objectives (Guides the AI)")
                
                if uploaded and st.button("🚀 Analyze Slides"):
                    st.info("Processing visual slides... please wait.")
                    
                    c.execute("INSERT INTO lectures (exam_id, name, raw_text) VALUES (?,?,?)", 
                              (e_map[sel_e], uploaded.name, "Visual"))
                    lid = c.lastrowid
                    conn.commit()
                    
                    images = process_file_visually(uploaded)
                    prog = st.progress(0)
                    
                    for i, img in enumerate(images):
                        cards = generate_cards_vision(img, objs)
                        for f, b in cards:
                            c.execute("INSERT INTO cards (lecture_id, front, back, next_review) VALUES (?,?,?,?)",
                                      (lid, f, b, date.today()))
                        conn.commit()
                        prog.progress((i+1)/len(images))
                        time.sleep(4) 
                        
                    st.success(f"Done! Processed {len(images)} slides.")

    with tab2:
        st.subheader("Manual Card Entry")
        if classes:
            # Re-select context
            c_map_m = {n: i for i, n in classes}
            sel_c_m = st.selectbox("Class", list(c_map_m.keys()), key="man_c")
            
            exams_m = c.execute("SELECT id, name FROM exams WHERE class_id=?", (c_map_m[sel_c_m],)).fetchall()
            if exams_m:
                e_map_m = {n: i for i, n in exams_m}
                sel_e_m = st.selectbox("Topic/Exam", list(e_map_m.keys()), key="man_e")
                
                # Fetch lectures or create a "Manual" bucket
                lecs = c.execute("SELECT id, name FROM lectures WHERE exam_id=?", (e_map_m[sel_e_m],)).fetchall()
                
                lid = None
                if not lecs:
                    if st.button("Create 'General' Bucket for this Topic"):
                        c.execute("INSERT INTO lectures (exam_id, name, raw_text) VALUES (?,?,?)",
                                  (e_map_m[sel_e_m], "General Manual Notes", ""))
                        conn.commit()
                        st.rerun()
                else:
                    l_map = {n: i for i, n in lecs}
                    sel_l = st.selectbox("Lecture Bucket", list(l_map.keys()))
                    lid = l_map[sel_l]
                
                if lid:
                    f_text = st.text_area("Front (Question)")
                    b_text = st.text_area("Back (Answer)")
                    if st.button("Add Card"):
                        c.execute("INSERT INTO cards (lecture_id, front, back, next_review) VALUES (?,?,?,?)",
                                  (lid, f_text, b_text, date.today()))
                        conn.commit()
                        st.success("Card Added!")
            else:
                st.info("Create a Topic in the 'AI Upload' tab first (you can do it without a date now!).")

# ------------------------------------------
# 3. CARD MANAGER (Edit/Delete)
# ------------------------------------------
elif menu == "Card Manager & Editor":
    st.header("🛠️ Card Editor")
    
    search = st.text_input("🔍 Search Cards (Drug name, concept...)")
    
    query = "SELECT id, front, back, next_review FROM cards"
    params = ()
    
    if search:
        query += " WHERE front LIKE ? OR back LIKE ?"
        params = (f'%{search}%', f'%{search}%')
        
    df = pd.read_sql(query, conn, params=params)
    
    if not df.empty:
        edited_df = st.data_editor(df, num_rows="dynamic", key="editor")
        
        if st.button("💾 Save Changes"):
            for index, row in edited_df.iterrows():
                c.execute("UPDATE cards SET front=?, back=? WHERE id=?", 
                          (row['front'], row['back'], row['id']))
            conn.commit()
            st.success("Database Updated!")
            
        st.markdown("---")
        del_id = st.number_input("ID to Delete", min_value=0, step=1)
        if st.button("🗑️ Delete Card"):
            c.execute("DELETE FROM cards WHERE id=?", (del_id,))
            conn.commit()
            st.warning(f"Deleted Card ID {del_id}")
            st.rerun()
            
# ------------------------------------------
# 4. BACKUP
# ------------------------------------------
elif menu == "Backup":
    st.header("💾 Backup Data")
    
    df = pd.read_sql("SELECT * FROM cards", conn)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        "Download Backup CSV",
        csv,
        "pharmpilot_backup.csv",
        "text/csv",
        key='download-csv'
    )

conn.close()