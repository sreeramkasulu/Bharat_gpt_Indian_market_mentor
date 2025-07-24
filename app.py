import os
import streamlit as st
import pdfplumber
import spacy
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 🔽 New line added to ensure deployment works (required by Streamlit Cloud)
import spacy.cli
spacy.cli.download("en_core_web_sm")

# Silence warnings (optional but helpful)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load SpaCy NLP
nlp = spacy.load("en_core_web_sm")

# ✅ Safe CPU loading (no .to() call)
retriever = SentenceTransformer("all-MiniLM-L6-v2")  # CPU by default

# ✅ Load LaMini model for generating answers
@st.cache_resource
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
    model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
    return tokenizer, model.to("cpu")

tokenizer, generator = load_generator()

# PDF Processing
PDF_PATH = "static/your_doc.pdf"

@st.cache_data(show_spinner=False)
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

sentences = extract_sentences(PDF_PATH)

# RAG Sentence Retriever
def get_relevant_sentences(question, threshold=0.5):
    all_text = [s for s, _ in sentences]
    q_embed = retriever.encode([question])
    s_embed = retriever.encode(all_text)
    scores = cosine_similarity(q_embed, s_embed)[0]
    matched = [(text, score) for (text, _), score in zip(sentences, scores) if score >= threshold]
    return [t for t, _ in sorted(matched, key=lambda x: -x[1])]

# Prompt with fallback
def is_question_answerable(question, context):
    prompt = (
        "You are an AI Assistant. Answer queries clearly only from the attached document. "
        "If the question is out of context, say \"I don't have an answer for this.\"\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = generator.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.strip()

# Streamlit UI
st.set_page_config(page_title="BharatGPT", page_icon="🤖")

st.markdown("""
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
""", unsafe_allow_html=True)

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

question = st.text_input("Type your question", placeholder="e.g. What is circular economy?")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("❗ Please enter a valid question.")
    else:
        with st.spinner("🔍 Searching..."):
            matched_sents = get_relevant_sentences(question)
            if matched_sents:
                context = " ".join(matched_sents[:5])
                verdict = is_question_answerable(question, context)
                st.markdown("### ✅ ANSWER:")
                if "I don't have an answer" in verdict:
                    st.error("❗No Answer: It's Out of Context")
                else:
                    for s in matched_sents:
                        st.success(s)
            else:
                st.markdown("### ✅ ANSWER:")
                st.error("❗No Answer: It's Out of Context")
