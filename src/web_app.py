# src/web_app.py
import streamlit as st
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader

# ---------------- Page config ----------------
st.set_page_config(
    page_title="NLP Question Generation — Student Learning & Feedback",
    layout="wide",
    page_icon="🧠"
)

# ---------------- NLTK setup ----------------
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_dir)

# ---------------- Utility functions ----------------
def read_pdf(filelike):
    text = ""
    try:
        reader = PdfReader(filelike)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
    return text

def read_txt(filelike):
    try:
        raw = filelike.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)
    except Exception as e:
        try:
            with open(filelike, "r", encoding="utf8") as fh:
                return fh.read()
        except Exception as ex:
            st.error(f"Could not read text file: {ex}")
    return ""

def chunk_text(text, chunk_size):
    sentences = sent_tokenize(text)
    chunks = [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size) if len(sentences[i:i+chunk_size])>0]
    return chunks

@st.cache_resource
def load_qg_model(name="t5-small"):
    tokenizer = T5Tokenizer.from_pretrained(name)
    model = T5ForConditionalGeneration.from_pretrained(name)
    model.eval()
    return tokenizer, model

def ensure_question_mark(q):
    q = q.strip()
    if not q:
        return q
    # if it already ends with ? or !, keep it
    if q.endswith("?") or q.endswith("!"):
        return q
    # If it looks like multiple sentences, take first sentence and ensure ?
    if "." in q and len(q.split(".")) > 1:
        first = q.split(".")[0].strip()
        return (first + "?").strip()
    return q + "?"

def generate_questions_from_text(text, tokenizer, model, max_questions=10, chunk_size=4):
    chunks = chunk_text(text, chunk_size)
    questions = []
    for chunk in chunks:
        prompt = f"generate question: {chunk}"
        try:
            enc = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
            outs = model.generate(enc, max_length=64, num_beams=4, early_stopping=True)
            q = tokenizer.decode(outs[0], skip_special_tokens=True).strip()
            q = ensure_question_mark(q)
            if q and q not in questions:
                questions.append(q)
        except Exception as e:
            st.warning(f"Question generation error for one chunk: {e}")
        if len(questions) >= max_questions:
            break
    return questions

# ---------------- Session state initialization ----------------
if "uploaded_text" not in st.session_state:
    st.session_state["uploaded_text"] = ""
if "questions" not in st.session_state:
    st.session_state["questions"] = []
if "answers" not in st.session_state:
    st.session_state["answers"] = []
if "score" not in st.session_state:
    st.session_state["score"] = 0
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False
if "model_name" not in st.session_state:
    st.session_state["model_name"] = "t5-small"

# ---------------- Header UI ----------------
st.markdown("""
<div style='background:linear-gradient(to right,#11998e,#38ef7d); padding:18px; border-radius:10px;'>
<h1 style='color:white; text-align:center; margin:0;'>NLP Question Generation — Student Learning & Feedback</h1>
<p style='color:white; text-align:center; margin:0;'>Upload a document or paste text, generate clear questions, submit answers, and get instant feedback.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar controls ----------------
st.sidebar.header("Options")
st.sidebar.markdown("Model & generation settings")
model_choice = st.sidebar.selectbox("Select QG Model", ["t5-small", "t5-base"], index=0)
chunk_size = st.sidebar.slider("Chunk size (sentences per chunk)", 2, 8, 4)
max_q = st.sidebar.slider("Max questions to generate", 1, 30, 10)
st.sidebar.markdown("---")
st.sidebar.write("Manage session")
if st.sidebar.button("Clear previous file & questions"):
    st.session_state["uploaded_text"] = ""
    st.session_state["questions"] = []
    st.session_state["answers"] = []
    st.session_state["score"] = 0
    st.session_state["submitted"] = False
    st.success("Cleared previous file and questions.")
if st.sidebar.button("Refresh / Regenerate (keep input)"):
    st.session_state["questions"] = []
    st.session_state["answers"] = []
    st.session_state["score"] = 0
    st.session_state["submitted"] = False
    st.experimental_rerun()
st.sidebar.markdown("---")
st.sidebar.markdown("Note: you can re-upload or paste new text anytime.")

# ---------------- Input area ----------------
st.subheader("Input (Paste text or Upload PDF / TXT)")
input_mode = st.radio("Input type:", ("Paste text", "Upload file"), horizontal=True)

user_text = ""
uploaded_file = None

if input_mode == "Paste text":
    user_text = st.text_area("Paste text here:", value=st.session_state.get("uploaded_text",""), height=200)
    if st.button("Use pasted text as input"):
        st.session_state["uploaded_text"] = user_text
        st.success("Text saved. Now click Generate Questions.")
elif input_mode == "Upload file":
    uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf","txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
            st.session_state["uploaded_text"] = read_pdf(uploaded_file)
        else:
            st.session_state["uploaded_text"] = read_txt(uploaded_file)
        st.success("File uploaded and text extracted. Now click Generate Questions.")

# ---------- Show current input preview ----------
if st.session_state.get("uploaded_text", "").strip():
    st.subheader("Document Preview")
    st.markdown(f"<div style='background:#f7f9f9;padding:12px;border-radius:8px;max-height:240px;overflow:auto'>{st.session_state['uploaded_text'][:5000]}</div>", unsafe_allow_html=True)
else:
    st.info("No input yet — paste text or upload a file and click the button to use it.")

# ---------------- Load model (cached) ----------------
with st.spinner("Loading model..."):
    try:
        tokenizer, model = load_qg_model(model_choice)
        st.session_state["model_name"] = model_choice
    except Exception as e:
        st.error(f"Could not load model {model_choice}: {e}")
        st.stop()

# ---------------- Generate questions ----------------
if st.button("Generate Questions"):
    text = st.session_state.get("uploaded_text", "").strip()
    if not text:
        st.warning("No input text. Paste or upload a file first.")
    else:
        with st.spinner("Generating questions..."):
            qs = generate_questions_from_text(text, tokenizer, model, max_questions=max_q, chunk_size=chunk_size)
        if not qs:
            st.error("No questions generated — try increasing chunk size or max questions.")
        else:
            st.session_state["questions"] = qs
            st.session_state["answers"] = [""] * len(qs)
            st.session_state["submitted"] = False
            st.success(f"Generated {len(qs)} questions.")

# ---------------- Show questions ----------------
if st.session_state.get("questions"):
    st.subheader("Generated Questions")
    for i, q in enumerate(st.session_state["questions"]):
        st.markdown(f"**Q{i+1}.** {q}")
        st.session_state["answers"][i] = st.text_input(f"Your answer for Q{i+1}", value=st.session_state["answers"][i], key=f"ans_{i}")

    # buttons for actions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Submit Answers"):
            # simple keyword-based feedback
            feedback = []
            score = 0
            for i, user_ans in enumerate(st.session_state["answers"]):
                q_text = st.session_state["questions"][i].lower()
                if user_ans and any(w for w in user_ans.lower().split() if w in q_text):
                    feedback.append((i, True))
                    score += 1
                else:
                    feedback.append((i, False))
            st.session_state["feedback"] = feedback
            st.session_state["score"] = score
            st.session_state["submitted"] = True
    with col2:
        if st.button("Clear Questions (keep input)"):
            st.session_state["questions"] = []
            st.session_state["answers"] = []
            st.session_state["score"] = 0
            st.session_state["submitted"] = False
            st.success("Questions cleared.")
    with col3:
        if st.button("Remove uploaded file & questions"):
            st.session_state["uploaded_text"] = ""
            st.session_state["questions"] = []
            st.session_state["answers"] = []
            st.session_state["score"] = 0
            st.session_state["submitted"] = False
            st.success("Removed uploaded file and questions. You may upload/paste new input.")

# ---------------- Feedback area ----------------
if st.session_state.get("submitted", False):
    st.subheader("Feedback & Score")
    for i, correct in st.session_state.get("feedback", []):
        if correct:
            st.success(f"Q{i+1}: Correct ✅")
        else:
            st.error(f"Q{i+1}: Incorrect ❌ — try checking keywords in the question.")
    total = len(st.session_state.get("questions", []))
    st.markdown(f"**Score:** {st.session_state.get('score',0)} / {total}")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("Made with ❤️ — NLP Question Generation: Student Learning & Feedback")
