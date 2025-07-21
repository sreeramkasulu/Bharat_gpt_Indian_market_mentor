import streamlit as st
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load NLP models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDF
PDF_PATH = "static/your_doc.pdf"
sentences = []

def extract_sentences(pdf_path):
    extracted = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                doc = nlp(text)
                for sent in doc.sents:
                    extracted.append((sent.text.strip(), page_number))
    return extracted

@st.cache_data(show_spinner=False)
def preload_sentences():
    return extract_sentences(PDF_PATH)

sentences = preload_sentences()

def find_best_sentences(question, threshold=0.5):
    sent_texts = [s for s, _ in sentences]
    question_emb = model.encode([question])
    sent_emb = model.encode(sent_texts)
    similarities = cosine_similarity(question_emb, sent_emb)[0]
    matched = [text for (text, _), score in zip(sentences, similarities) if score >= threshold]
    return matched if matched else ["❗No Answer: It's Out of Context"]

# 🌐 UI STARTS HERE
st.set_page_config(page_title="BharatGPT", page_icon="🤖")
st.markdown(
    """
    <style>
    body { background: linear-gradient(to bottom right, #f0f9ff, #d9f0e1); }
    .stTextInput > div > div > input {
        font-size: 16px;
        padding: 10px;
        border-radius: 6px;
    }
    .stButton > button {
        background-color: #2196f3;
        color: white;
        border-radius: 6px;
        padding: 8px 20px;
        font-size: 15px;
    }
    .stButton > button:hover {
        background-color: #1976d2;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("💬 BharatGPT — Your Indian Market Mentor 🤖")

with st.expander("💡 What Can You Ask?", expanded=True):
    st.markdown("""
    Ask BharatGPT about:
    - 📦 India’s rules for biodegradable packaging  
    - 🏭 Setting up a recycling facility with ULBs  
    - 📊 Government schemes for sustainability startups  
    - 🧪 Partnering with IISc, CSIR, or BIRAC  
    - 💡 Examples like MYNUSCo, Recykal, or BioE3 Policy  
    """)

st.markdown("## ")

question = st.text_input("Type your question", placeholder="e.g. What is circular economy?")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("❗ Please enter a valid question.")
    else:
        with st.spinner("🔍 Searching..."):
            answers = find_best_sentences(question)
        st.markdown("### ✅ ANSWER:")
        for ans in answers:
            st.success(ans)
