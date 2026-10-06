import os
import shutil
import tempfile
import uuid
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from groq import NotFoundError

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------
# Page Config & Custom Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Nexus RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 0%, rgba(147, 197, 253, 0.38), transparent 30%),
                    radial-gradient(circle at 90% 90%, rgba(191, 219, 254, 0.6), transparent 35%),
                    #eff6ff;
        color: #172554;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dbeafe 0%, #bfdbfe 45%, #93c5fd 100%) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(59, 130, 246, 0.25) !important;
        box-shadow: 4px 0 25px rgba(96, 165, 250, 0.25);
    }
    [data-testid="stSidebar"] > div {
        background: transparent;
    }
    [data-testid="stSidebar"], [data-testid="stSidebar"] * {
        color: #172554;
    }

    .hero-card {
        padding: 2.2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 55%, #ffedd5 100%);
        border: 1px solid #fed7aa;
        box-shadow: 0 10px 30px -10px rgba(234, 138, 62, 0.18);
        margin-bottom: 1.8rem;
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 55%, #ea7a28 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .hero-sub {
        color: #475569;
        font-size: 1.1rem;
        font-weight: 400;
        line-height: 1.6;
    }

    .stat-card {
        background: linear-gradient(145deg, #ffffff 0%, #eff6ff 68%, #fff7ed 100%);
        border: 1px solid #fed7aa;
        border-radius: 20px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(234, 138, 62, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: #fb923c;
    }

    .stat-val {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2563eb, #ea7a28);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-lbl {
        font-size: 0.82rem;
        color: #475569;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    [data-testid="stAppViewContainer"] {
        color: #172554;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
        background-color: #ffffff;
        color: #172554;
        border-color: #93c5fd;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 1px #2563eb;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb, #ea7a28);
        color: #ffffff;
        border: 0;
    }
    .stButton > button[kind="secondary"] {
        border-color: #fdba74;
        color: #9a3412;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Helper Functions & Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Embedding model load ho raha hai...")
def load_embeddings():
    """Load HuggingFace Embedding Model once."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_db(uploaded_files, chunk_size, overlap):
    """Process uploaded PDFs, split text, and store in Chroma vectorstore."""
    all_docs = []

    for pdf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.getvalue())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for page in pages:
                page.metadata["source_file"] = pdf.name
            all_docs.extend(pages)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = text_splitter.split_documents(all_docs)

    # Persist directory setting to avoid vector loss on streamlit rerun
    persist_dir = os.path.join(tempfile.gettempdir(), f"chroma_db_{uuid.uuid4().hex[:8]}")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=load_embeddings(),
        persist_directory=persist_dir
    )

    return vectorstore, all_docs, chunks, persist_dir


def get_context_string(docs):
    """Format retrieved document chunks into formatted context."""
    formatted_chunks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "Unknown File")
        page = doc.metadata.get("page", 0) + 1
        formatted_chunks.append(
            f"[Source {idx} | File: {source} | Page: {page}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted_chunks)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
default_states = {
    "vectorstore": None,
    "persist_dir": None,
    "pages": [],
    "chunks": [],
    "file_names": [],
    "last_question": "",
    "last_answer": "",
    "last_evidence": [],
}

for k, v in default_states.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚡ Nexus Controls")

    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        help="Paste your API key here or set GROQ_API_KEY in .env",
    )
    groq_api_key = (api_key_input or os.getenv("GROQ_API_KEY") or "").strip()

    st.subheader("Model Configuration")
    supported_groq_models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ]
    model_name = st.selectbox(
        "Select LLM Model",
        supported_groq_models,
        index=0,
    )
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.2, 0.05)
    strict_mode = st.checkbox("Strict Document Grounding", value=True)

    st.subheader("Retrieval Settings")
    search_strategy = st.selectbox(
        "Search Mode", ["MMR (Diverse Results)", "Similarity (Closest Match)"]
    )
    top_k = st.slider("Top Chunks (k)", 2, 8, 4)

    st.subheader("Chunking Parameters")
    chunk_size = st.slider("Chunk Size", 300, 1500, 800, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 300, 100, 10)

    if st.button("Reset Session", use_container_width=True):
        if st.session_state.persist_dir and os.path.exists(st.session_state.persist_dir):
            shutil.rmtree(st.session_state.persist_dir, ignore_errors=True)
        for k in default_states:
            st.session_state[k] = default_states[k]
        st.rerun()


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">⚡ Nexus RAG Studio</div>
    <div class="hero-sub">Transform documents into an intelligent knowledge base with real-time vector search, grounding, and precise source citations.</div>
</div>
""",
    unsafe_allow_html=True,
)

if not groq_api_key:
    st.warning("⚠️ Please provide a valid Groq API Key in the sidebar or `.env` file to proceed.")
    st.stop()

# Initialize LLM
try:
    llm = ChatGroq(
        api_key=groq_api_key,
        model=model_name,
        temperature=temperature,
    )
except Exception as exc:
    st.error(f"Unable to initialize Groq model: {exc}")
    st.stop()


def invoke_llm(prompt):
    try:
        return llm.invoke(prompt)
    except NotFoundError:
        st.error(
            f"Groq does not provide access to the selected model '{model_name}'. "
            "Choose another model from the sidebar and try again."
        )
        return None


# ---------------------------------------------------------
# Document Upload & Knowledge Base Construction
# ---------------------------------------------------------
with st.expander("📚 Knowledge Base Setup / Upload Documents", expanded=(st.session_state.vectorstore is None)):
    uploaded_files = st.file_uploader(
        "Upload one or multiple PDF files", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Build Knowledge Base", type="primary"):
        if not uploaded_files:
            st.error("Please select at least one PDF file.")
        else:
            with st.spinner("Processing PDFs, creating embeddings, and indexing..."):
                vstore, pages, chunks, persist_dir = build_vector_db(
                    uploaded_files, chunk_size, chunk_overlap
                )
                st.session_state.vectorstore = vstore
                st.session_state.persist_dir = persist_dir
                st.session_state.pages = pages
                st.session_state.chunks = chunks
                st.session_state.file_names = [f.name for f in uploaded_files]
                st.session_state.last_answer = ""
                st.session_state.last_evidence = []

            st.success("Knowledge Base created successfully!")
            st.rerun()

if st.session_state.vectorstore is None:
    st.info("👈 Upload PDF documents above to activate the RAG studio.")
    st.stop()


# ---------------------------------------------------------
# Application Tabs
# ---------------------------------------------------------
tab_dash, tab_ask, tab_search, tab_study = st.tabs(
    ["📊 Dashboard", "💬 Ask & Evidence", "🔍 Vector Search", "🎓 Study Assistant"]
)

# --- TAB 1: DASHBOARD ---
with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="stat-card"><div class="stat-val">{len(st.session_state.file_names)}</div><div class="stat-lbl">Documents</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card"><div class="stat-val">{len(st.session_state.pages)}</div><div class="stat-lbl">Total Pages</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-card"><div class="stat-val">{len(st.session_state.chunks)}</div><div class="stat-lbl">Total Chunks</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="stat-card"><div class="stat-val">{top_k}</div><div class="stat-lbl">Top-K Chunks</div></div>', unsafe_allow_html=True)

    st.markdown("### 📁 Uploaded Files Summary")
    for f_name in st.session_state.file_names:
        p_count = sum(1 for p in st.session_state.pages if p.metadata.get("source_file") == f_name)
        c_count = sum(1 for c in st.session_state.chunks if c.metadata.get("source_file") == f_name)
        st.write(f"📄 **{f_name}** — {p_count} Pages | {c_count} Chunks")

# --- TAB 2: ASK & EVIDENCE ---
with tab_ask:
    st.markdown("### 💬 Ask Questions Against Documents")
    query = st.text_input("Enter your question:", value=st.session_state.last_question)

    if st.button("Generate Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant context and generating answer..."):
                search_type = "mmr" if "MMR" in search_strategy else "similarity"
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type=search_type, search_kwargs={"k": top_k}
                )
                docs = retriever.invoke(query)
                context_str = get_context_string(docs)

                system_prompt = (
                    "You are a helpful assistant. Use ONLY the provided context to answer the question. "
                    "If the answer cannot be found in the context, state that clearly."
                    if strict_mode
                    else "Answer the question using the context provided below. Supplement with general knowledge only if required."
                )

                prompt = f"{system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
                response = invoke_llm(prompt)

                if response is not None:
                    st.session_state.last_question = query
                    st.session_state.last_answer = response.content
                    st.session_state.last_evidence = docs

    if st.session_state.last_answer:
        st.markdown("### 🤖 Response")
        st.markdown(st.session_state.last_answer)

        st.markdown("### 🔍 Retrieved Evidence")
        for idx, doc in enumerate(st.session_state.last_evidence, 1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", 0) + 1
            with st.expander(f"Chunk {idx} | File: {source} (Page {page})"):
                st.write(doc.page_content)

# --- TAB 3: VECTOR SEARCH ---
with tab_search:
    st.markdown("### 🔍 Raw Vector Search Exploration")
    search_query = st.text_input("Query vector DB:")
    search_k = st.slider("Results to show", 1, 10, 4)

    if search_query:
        docs = st.session_state.vectorstore.similarity_search(search_query, k=search_k)
        for idx, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", 0) + 1
            st.markdown(f"**Result {idx}** (`{source}`, Page {page})")
            st.info(doc.page_content)

# --- TAB 4: STUDY ASSISTANT ---
with tab_study:
    st.markdown("### 🎓 Study & Review Assistant")
    study_action = st.radio("Select Tool:", ["Generate Summary", "Create Practice Quiz", "Extract Key Terms"])

    if st.button("Run Study Tool", type="primary"):
        sample_chunks = st.session_state.chunks[:min(10, len(st.session_state.chunks))]
        context_str = get_context_string(sample_chunks)

        if study_action == "Generate Summary":
            prompt = f"Summarize the core topics and essential findings based on this context:\n\n{context_str}"
        elif study_action == "Create Practice Quiz":
            prompt = f"Create a 5-question multiple-choice quiz with answers based on this context:\n\n{context_str}"
        else:
            prompt = f"Extract and define the key terms and concepts from this context:\n\n{context_str}"

        with st.spinner("Processing request..."):
            response = invoke_llm(prompt)
            if response is not None:
                st.markdown(response.content)