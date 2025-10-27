import streamlit as st
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader

# ---------------- NLTK Setup ----------------
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_dir)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

# ---------------- Utility Functions ----------------
def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def read_txt(path):
    with open(path, 'r', encoding='utf8') as f:
        return f.read()

def chunk_text(text, chunk_size=5):
    sentences = sent_tokenize(text)
    chunks = [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

@st.cache_resource
def load_model(model_name='t5-base'):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def generate_questions(text, tokenizer, model, max_questions=10):
    chunks = chunk_text(text)
    questions = []
    for c in chunks:
        input_text = "generate a meaningful question with a question mark: " + c
        encoding = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
        outputs = model.generate(
            encoding,
            max_length=64,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        q = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if q and q.endswith("?") and q not in questions:
            questions.append(q)
        if len(questions) >= max_questions:
            break
    return questions

# ---------------- Streamlit Layout ----------------
st.set_page_config(
    page_title="🎓 NLP Question Generation & Student Feedback",
    layout="wide",
    page_icon="🧠"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
body {
    color: #000000;
}
.card {
    background-color: #fff8eb;
    color: #000000;
    padding: 18px;
    margin-bottom: 15px;
    border-radius: 15px;
    border-left: 6px solid #f39c12;
    box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
}
.scrollable-area {
    max-height: 300px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 10px;
    background-color: #fdfdfd;
    color: #000000;
    font-size: 16px;
}
.correct {
    background-color: #d4edda;
    color: #155724;
}
.incorrect {
    background-color: #f8d7da;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div style='background: linear-gradient(to right, #ff9966, #ff5e62); padding: 25px; border-radius: 15px; text-align:center; color:white;'>
<h1>🎓 NLP Question Generation & Student Feedback App</h1>
<p>Upload a document or paste text to generate meaningful, well-formed questions automatically.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Select Model", ['t5-small','t5-base'])
chunk_size = st.sidebar.slider("Chunk Size (sentences per chunk)", 1, 10, 5)
max_q = st.sidebar.slider("Max Questions", 1, 20, 10)
st.sidebar.markdown("---")

# ---------- Input Options ----------
input_option = st.radio("Choose Input Type", ["Paste Text", "Upload File"], horizontal=True)

input_text, file_text = "", ""

# Add a refresh button
if st.button("🔄 Clear All / Start Fresh"):
    st.session_state.clear()
    st.experimental_rerun()

if input_option == "Paste Text":
    input_text = st.text_area("✏️ Paste your text here", height=200, placeholder="Type or paste content here...")
elif input_option == "Upload File":
    uploaded_file = st.file_uploader("📂 Upload PDF or TXT file", type=['pdf', 'txt'])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            file_text = read_pdf(uploaded_file)
        else:
            file_text = read_txt(uploaded_file)

# ---------- Merge Input ----------
combined_text = input_text.strip() or file_text.strip()

if combined_text:
    st.subheader("📄 Document Preview")
    st.markdown(f"<div class='scrollable-area'>{combined_text}</div>", unsafe_allow_html=True)

# ---------- Load model ----------
with st.spinner("🔍 Loading model... please wait"):
    tokenizer, model = load_model(model_choice)

# ---------- Generate Questions ----------
if st.button("🚀 Generate Meaningful Questions"):
    if not combined_text:
        st.warning("Please provide text or upload a document first!")
    else:
        with st.spinner("🧠 Generating well-structured questions..."):
            questions = generate_questions(combined_text, tokenizer, model, max_questions=max_q)
        if not questions:
            st.error("❌ No meaningful questions generated. Try increasing the chunk size or input length.")
        else:
            st.session_state['questions'] = questions
            st.success(f"✅ Generated {len(questions)} professional questions!")

# ---------- Display Questions ----------
if 'questions' in st.session_state:
    st.subheader("📝 Generated Questions")
    for i, q in enumerate(st.session_state['questions']):
        st.markdown(f"<div class='card'><b>Q{i+1}:</b> {q}</div>", unsafe_allow_html=True)
