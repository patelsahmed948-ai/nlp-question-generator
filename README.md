# NLP Project — Question Generator & Student Feedback

A Streamlit web app that allows students to upload PDF/TXT documents, auto-generates questions using T5, extracts predicted answers, and gives immediate feedback on student answers.

## Quick setup
1. Create & activate the virtual environment:
   - python -m venv venv
   - env\Scripts\activate

2. Install dependencies:
   - pip install -r requirements.txt
   - If torch install fails, use the CPU wheel:
     pip install torch --index-url https://download.pytorch.org/whl/cpu

3. Run the app:
   - streamlit run src\web_app.py

## Folder structure
- data/      : sample inputs and saved outputs
- models/    : place a fine-tuned model 	5_qg.pt here (optional)
- src/       : source code (app, utils, optional training scripts)
- env/      : Python virtual environment

## Features
- Upload PDF/TXT
- Split document into chunks and generate questions (T5)
- Optional QA answer extraction (DistilBERT SQuAD)
- Student answer input with fuzzy-match feedback
- Save generated Q&A to CSV / JSON

## Notes
- Download NLTK data (punkt): python -c "import nltk; nltk.download('punkt')"
- If PDFs are scanned images, consider adding OCR (Tesseract) later.
