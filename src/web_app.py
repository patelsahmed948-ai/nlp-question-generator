# src/web_app.py
import streamlit as st
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader

# ---------------- NLTK Setup ----------------
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

# ---------------- Utility Functions ----------------
def read_pdf(f):
    """
    Accepts either a file path (str) or a file-like object (Streamlit uploaded file).
    """
    text = ""
    try:
        # PdfReader can accept file-like objects returned by Streamlit uploaded_file
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        # Fallback: if f is path string
        try:
            with open(f, "rb") as fh:
                reader = PdfReader(fh)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception:
            st.error(f"Could not read PDF: {e}")
    return text

def read_txt(f):
    """
    Accepts either a file path (str) or a file-like object (Streamlit uploaded file).
    """
    try:
        if hasattr(f, "read"):
            raw = f.read()
            # uploaded_file.read() returns bytes for Streamlit file uploader
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="ignore")
            return str(raw)
        else:
            with open(f, "r", encoding="utf8") as fh:
                return fh.read()
    except Exception as e:
        st.error(f"Could not read text file: {e}")
        return ""

def chunk_text(text, chunk_size=5):
    sentences = sent_tokenize(text)
    chunks = [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size) if len(sentences[i:i+chunk_size])>0]
    return chunks

@st.cache_resource
def load_model(model_name='t5-small'):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model '{model_name}': {e}")
        raise

def generate_questions(text, tokenizer, model, max_questions=10, chunk_size=5):
    chunks = chunk_text(text, chunk_size=chunk_size)
    questions = []
    for c in chunks:
        # small safety: skip very short chunks
        if len(c.split()) < 5:
            continue
        input_text = "generate question: " + c  # keep prompt short and consistent
        try:
            encoding = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
            outputs = model.generate(encoding, max_length=64, num_beams=4, early_stopping=True)
            q = tokenizer.decode(outputs[0], skip_special_tokens=True)
            q = q.strip()
            if q:
                questions.append(q)
        except Exception as e:
            # continue on generation error, but show once
            st.warning(f"Generation error for one chunk: {e}")
        if len(questions) >= max_questions:
            break
    return questions

# ---------------- Streamlit Layout ----------------
st.set_page_config(page_title="📝 NLP Quiz Generator", layout="wide", page_icon="📝")

# ---------- Custom CSS ----------
st.markdown("""
<style>
.card {
    background-color: #fff4e6;  /* card background */
    color: #000000;             /* force black text */
    padding: 20px;
    margin-bottom: 15px;
    border-radius: 15px;
    border-left: 6px solid #ff7f50;
    box-shadow: 3px 3px 12px #ccc;
}
.correct {
    background-color: #d4edda !important;
    color: #155724 !important;
}
.incorrect {
    background-color: #f8d7da !important;
    color: #721c24 !important;
}
.scrollable-area {
    max-height: 300px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 10px;
    background-color: #fafafa;
    color: #000000;  /* force black text */
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div style='background: linear-gradient(to right, #f7971e, #ffd200); padding: 25px; border-radius: 15px;'>
<h1 style='text-align:center; color:white;'>📝 Modern NLP Quiz Generator</h1>
<p style='text-align:center; color:white;'>Paste text or upload PDF/TXT to generate interactive questions!</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.header("Settings & Options")
model_choice = st.sidebar.selectbox("Select Model", ['t5-small','t5-base'])
chunk_size = st.sidebar.slider("Chunk Size (sentences per chunk)", 1, 10, 5)
max_q = st.sidebar.slider("Max Questions", 1, 20, 10)
st.sidebar.markdown("---")

# ---------- Input Options ----------
input_option = st.radio("Choose Input Type", ["Paste Text", "Upload File"], horizontal=True)

# Ensure session state keys exist
if "input_text" not in st.session_state:
    st.session_state['input_text'] = ""
if "file_text" not in st.session_state:
    st.session_state['file_text'] = ""
if "questions" not in st.session_state:
    st.session_state['questions'] = []
if "answers" not in st.session_state:
    st.session_state['answers'] = []
if "submitted" not in st.session_state:
    st.session_state['submitted'] = False

input_text = ""
file_text = ""

if input_option == "Paste Text":
    if st.button("Clear Text"):
        st.session_state['input_text'] = ""
    input_text = st.text_area("Paste your text here", height=200, placeholder="Type or paste text here...", value=st.session_state.get('input_text', ''))
    st.session_state['input_text'] = input_text

elif input_option == "Upload File":
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=['pdf','txt'])
    if st.button("Clear File"):
        # clearing upload isn't straightforward; reset stored content
        st.session_state['file_text'] = ""
        uploaded_file = None
    if uploaded_file:
        if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
            file_text = read_pdf(uploaded_file)
        else:
            # for text files use getvalue
            try:
                raw = uploaded_file.getvalue()
                if isinstance(raw, bytes):
                    file_text = raw.decode("utf-8", errors="ignore")
                else:
                    file_text = str(raw)
            except Exception:
                file_text = read_txt(uploaded_file)
        st.session_state['file_text'] = file_text

# ---------- Merge Input ----------
combined_text = ""
if st.session_state.get('input_text', "").strip() and st.session_state.get('file_text', "").strip():
    combined_text = st.session_state['file_text'] + "\n" + st.session_state['input_text']
elif st.session_state.get('input_text', "").strip():
    combined_text = st.session_state['input_text']
elif st.session_state.get('file_text', "").strip():
    combined_text = st.session_state['file_text']

if combined_text.strip():
    st.subheader("📄 Full Text Preview")
    st.markdown(f"<div class='scrollable-area'>{combined_text}</div>", unsafe_allow_html=True)

# ---------- Load model ----------
with st.spinner("Loading model..."):
    try:
        tokenizer, model = load_model(model_choice)
    except Exception:
        st.stop()

# ---------- Generate Questions ----------
if st.button("Generate Questions"):
    if not combined_text.strip():
        st.warning("Please provide text or upload a file!")
    else:
        with st.spinner("Generating questions..."):
            questions = generate_questions(combined_text, tokenizer, model, max_questions=max_q, chunk_size=chunk_size)
        if len(questions) == 0:
            st.error("❌ No questions generated. Try increasing chunk size or max questions.")
        else:
            st.success(f"✅ {len(questions)} Questions Generated!")
            st.session_state['questions'] = questions
            st.session_state['answers'] = [""]*len(questions)
            st.session_state['submitted'] = False

# ---------- Display Questions in Stylish Cards ----------
if st.session_state.get('questions'):
    st.subheader("📝 Questions")
    answers = st.session_state.get('answers', [""]*len(st.session_state['questions']))
    for i, q in enumerate(st.session_state['questions']):
        st.markdown(f"<div class='card'><b>Q{i+1}:</b> {q}</div>", unsafe_allow_html=True)
        answers[i] = st.text_input(f"Your Answer for Question {i+1}", value=answers[i], key=f"ans{i}")
    st.session_state['answers'] = answers

    # ---------- Submit Button ----------
    if st.button("Submit All Answers"):
        st.session_state['submitted'] = True
        st.session_state['feedback'] = []
        score = 0
        for i, user_ans in enumerate(st.session_state['answers']):
            question = st.session_state['questions'][i].lower()
            if user_ans and any(word for word in user_ans.lower().split() if word in question):
                st.session_state['feedback'].append((f"Correct ✅", i))
                score += 1
            else:
                st.session_state['feedback'].append((f"Incorrect ❌. Hint: check key terms in the question.", i))
        st.session_state['score'] = score

# ---------- Show Feedback & Score ----------
if st.session_state.get('submitted', False):
    st.subheader("✅ Feedback")
    for msg, i in st.session_state.get('feedback', []):
        color_class = "correct" if "Correct" in msg else "incorrect"
        st.markdown(f"<div class='{color_class} card'>Q{i+1}: {msg}</div>", unsafe_allow_html=True)
    total = len(st.session_state.get('questions', []))
    st.markdown(f"<h3 style='color:#2E86C1'>Your Score: {st.session_state.get('score',0)} / {total}</h3>", unsafe_allow_html=True)
