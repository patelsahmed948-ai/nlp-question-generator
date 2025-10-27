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
def load_model(model_name='t5-small'):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def generate_questions(text, tokenizer, model, max_questions=10):
    chunks = chunk_text(text)
    questions = []
    for c in chunks:
        input_text = "generate questions: " + c
        encoding = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(encoding, max_length=64, num_beams=4, early_stopping=True)
        q = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if q.strip() != "":
            questions.append(q)
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
        uploaded_file = None
        st.session_state['file_text'] = ""
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            file_text = read_pdf(uploaded_file)
        else:
            file_text = read_txt(uploaded_file)
        st.session_state['file_text'] = file_text

# ---------- Merge Input ----------
combined_text = ""
if input_text.strip() and file_text.strip():
    combined_text = file_text + "\n" + input_text
elif input_text.strip():
    combined_text = input_text
elif file_text.strip():
    combined_text = file_text

if combined_text.strip():
    st.subheader("📄 Full Text Preview")
    st.markdown(f"<div class='scrollable-area'>{combined_text}</div>", unsafe_allow_html=True)

# ---------- Load model ----------
with st.spinner("Loading model..."):
    tokenizer, model = load_model(model_choice)

# ---------- Generate Questions ----------
if st.button("Generate Questions"):
    if not combined_text.strip():
        st.warning("Please provide text or upload a file!")
    else:
        with st.spinner("Generating questions..."):
            questions = generate_questions(combined_text, tokenizer, model, max_questions=max_q)
        if len(questions) == 0:
            st.error("❌ No questions generated. Try increasing chunk size or max questions.")
        else:
            st.success(f"✅ {len(questions)} Questions Generated!")
            st.session_state['questions'] = questions
            st.session_state['answers'] = [""]*len(questions)
            st.session_state['submitted'] = False

# ---------- Display Questions in Stylish Cards ----------
if 'questions' in st.session_state:
    st.subheader("📝 Questions")
    answers = st.session_state['answers']
    for i, q in enumerate(st.session_state['questions']):
        st.markdown(f"<div class='card'><b>Q{i+1}:</b> {q}</div>", unsafe_allow_html=True)
        answers[i] = st.text_input(f"Your Answer for Question {i+1}", value=answers[i], key=f"ans{i}")

    # ---------- Submit Button ----------
    if st.button("Submit All Answers"):
        st.session_state['submitted'] = True
        st.session_state['feedback'] = []
        score = 0
        for i, user_ans in enumerate(answers):
            question = st.session_state['questions'][i].lower()
            if any(word in question for word in user_ans.lower().split()) and user_ans.strip() != "":
                st.session_state['feedback'].append(("Correct ✅", i))
                score += 1
            else:
                st.session_state['feedback'].append(("Incorrect ❌. Hint: check key terms in the question.", i))
        st.session_state['score'] = score

# ---------- Show Feedback & Score ----------
if st.session_state.get('submitted', False):
    st.subheader("✅ Feedback")
    for msg, i in st.session_state['feedback']:
        color_class = "correct" if "Correct" in msg else "incorrect"
        st.markdown(f"<div class='{color_class} card'>Q{i+1}: {msg}</div>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:#2E86C1'>Your Score: {st.session_state['score']} / {len(st.session_state['questions'])}</h3>", unsafe_allow_html=True)
