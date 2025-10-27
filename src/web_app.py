import streamlit as st
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader

# ------------------- NLTK Setup -------------------
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_dir)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

# ------------------- Utility Functions -------------------
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

def chunk_text(text, chunk_size=4):
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
        input_text = f"Generate educational quiz questions for students from the following text: {c}"
        encoding = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
        outputs = model.generate(
            encoding,
            max_length=80,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        q = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if not q.endswith("?"):
            q += "?"
        if q not in questions:
            questions.append(q)
        if len(questions) >= max_questions:
            break
    return questions

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(
    page_title="🎓 NLP Student Quiz Generator",
    layout="wide",
    page_icon="🧠"
)

# ------------------- Modern CSS Styling -------------------
st.markdown("""
<style>
body {
    color: #000000;
    background: #f9fafc;
    font-family: 'Segoe UI', sans-serif;
}
.header {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.button-style button {
    background-color: #0078ff !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: none !important;
}
.card {
    background-color: white;
    color: #000000;
    padding: 20px;
    margin: 10px 0;
    border-radius: 15px;
    border-left: 6px solid #4facfe;
    box-shadow: 3px 3px 12px rgba(0,0,0,0.1);
}
.feedback {
    background-color: #e9f7ef;
    color: #155724;
    padding: 10px;
    border-radius: 10px;
    margin-top: 10px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ------------------- Header -------------------
st.markdown("""
<div class='header'>
<h1>🎓 NLP Question Generation & Student Learning Feedback</h1>
<p>Upload content or paste text to automatically generate student-friendly quiz questions.</p>
</div>
""", unsafe_allow_html=True)

# ------------------- Sidebar Settings -------------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["t5-small", "t5-base"])
chunk_size = st.sidebar.slider("Chunk Size (sentences per chunk)", 1, 10, 4)
max_q = st.sidebar.slider("Max Questions", 1, 20, 10)
st.sidebar.markdown("---")

# ------------------- Input Options -------------------
input_option = st.radio("Choose Input Method", ["Paste Text", "Upload File"], horizontal=True)
input_text, file_text = "", ""

# Proper reset logic
if st.button("🔄 Clear All & Restart"):
    st.cache_resource.clear()
    st.session_state.clear()
    st.rerun()

if input_option == "Paste Text":
    input_text = st.text_area("✏️ Paste your study material here", height=200, placeholder="Type or paste your notes or content here...")
elif input_option == "Upload File":
    uploaded_file = st.file_uploader("📂 Upload PDF or TXT file", type=['pdf', 'txt'])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            file_text = read_pdf(uploaded_file)
        else:
            file_text = read_txt(uploaded_file)

# ------------------- Merge Input -------------------
combined_text = input_text.strip() or file_text.strip()

# ------------------- Load model -------------------
with st.spinner("🧠 Loading NLP Model... Please wait"):
    tokenizer, model = load_model(model_choice)

# ------------------- Generate Questions -------------------
if st.button("🚀 Generate Questions", key="generate_btn"):
    if not combined_text:
        st.warning("Please provide text or upload a file before generating questions.")
    else:
        with st.spinner("🧩 Creating student-friendly quiz questions..."):
            questions = generate_questions(combined_text, tokenizer, model, max_questions=max_q)
        if not questions:
            st.error("⚠️ No questions could be generated. Try providing more text or increasing chunk size.")
        else:
            st.success(f"✅ {len(questions)} questions generated successfully!")
            st.session_state['questions'] = questions
            st.session_state['answers'] = [""] * len(questions)
            st.session_state['submitted'] = False

# ------------------- Display Questions -------------------
if 'questions' in st.session_state:
    st.subheader("📝 Generated Quiz Questions")
    answers = st.session_state['answers']

    for i, q in enumerate(st.session_state['questions']):
        st.markdown(f"<div class='card'><b>Q{i+1}:</b> {q}</div>", unsafe_allow_html=True)
        answers[i] = st.text_input(f"Your Answer for Question {i+1}", value=answers[i], key=f"ans{i}")

    # Submit button
    if st.button("📤 Submit Answers"):
        st.session_state['submitted'] = True
        score = 0
        feedback_msgs = []
        for i, ans in enumerate(answers):
            if len(ans.strip()) > 3:
                feedback_msgs.append(f"✅ Q{i+1}: Great effort! Your answer is noted.")
                score += 1
            else:
                feedback_msgs.append(f"❌ Q{i+1}: Try providing a more detailed answer.")
        st.session_state['feedback'] = feedback_msgs
        st.session_state['score'] = score

# ------------------- Feedback Section -------------------
if st.session_state.get('submitted', False):
    st.subheader("📊 Feedback & Results")
    for msg in st.session_state['feedback']:
        st.markdown(f"<div class='feedback'>{msg}</div>", unsafe_allow_html=True)
    total = len(st.session_state['questions'])
    st.markdown(f"<h3 style='color:#0078ff;'>Final Score: {st.session_state['score']} / {total}</h3>", unsafe_allow_html=True)
