import os
import streamlit as st
import pdfplumber
import spacy
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Streamlit UI Setup
st.set_page_config(page_title="BharatGPT", page_icon="🤖")
st.title("💬 BharatGPT — Your Indian Market Mentor 🤖")

# Load PDF
PDF_PATH = "static/your_doc.pdf"

@st.cache_data(show_spinner=False)
def load_docs(pdf_path):
    loader = PyPDFLoader(pdf_path)
    raw_pages = loader.load_and_split()
    return raw_pages

docs = load_docs(PDF_PATH)

# Sentence-level splitting
@st.cache_data(show_spinner=False)
def split_sentences(_docs):
    splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)
    return splitter.split_documents(_docs)

split_docs = split_sentences(docs)


# Embedding model for retrieval
@st.cache_resource
def build_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(split_docs, embedding_model)
    return db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

retriever = build_retriever()

# Prompt (just returning matched sentences)
prompt_template = PromptTemplate.from_template("""
You are BharatGPT, a strict assistant that only answers from the attached document. 
Return only the original sentence(s) from the document that match the question. Do not rephrase, summarize, or guess.

If nothing matches, say: "I don't have an answer for this."

Context: {context}
Question: {question}
Answer:
""")

# Load HuggingFace T5 model
@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
    tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_model()

# RAG chain setup
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# Sidebar instructions
with st.expander("💡 What Can You Ask?", expanded=True):
    st.markdown("""
    Ask BharatGPT about:
    - 📦 India’s rules for biodegradable packaging  
    - 🏭 Setting up a recycling facility with ULBs  
    - 📊 Government schemes for sustainability startups  
    - 🧪 Partnering with IISc, CSIR, or BIRAC  
    - 💡 Examples like MYNUSCo, Recykal, or BioE3 Policy  
    """)

# Ask a question
question = st.text_input("Type your question", placeholder="e.g. What is circular economy?")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("❗ Please enter a valid question.")
    else:
        with st.spinner("🔍 Searching..."):
            result = rag_chain({"query": question})
            response = result["result"]
            if "I don't have an answer" in response:
                st.error("❗ No Answer: It's Out of Context")
            else:
                st.markdown("### ✅ Matched Sentences:")
                st.success(response.strip())
