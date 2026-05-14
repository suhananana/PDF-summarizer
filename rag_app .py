import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}
.stApp {
    background: #f7f5f0;
}
section[data-testid="stSidebar"] {
    background: #1a1a2e;
    color: #e0e0e0;
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
.chat-bubble-user {
    background: #1a1a2e;
    color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 18px;
    margin: 6px 0;
    max-width: 75%;
    float: right;
    clear: both;
}
.chat-bubble-ai {
    background: #fff;
    color: #1a1a2e;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 18px;
    margin: 6px 0;
    max-width: 80%;
    float: left;
    clear: both;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.clearfix::after { content: ""; display: table; clear: both; }
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
}
.badge-ready { background: #d4edda; color: #155724; }
.badge-waiting { background: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "chunks": None,
    "embeddings": None,
    "embedding_model": None,
    "gemini_model": None,
    "pdf_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    return "\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - overlap
    return chunks


def build_index(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=False).astype("float32")
    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)
    return embeddings


def retrieve(query: str, top_k: int = 5):
    emb = st.session_state.embedding_model.encode([query]).astype("float32")
    emb = emb / np.maximum(np.linalg.norm(emb), 1e-9)
    scores = st.session_state.embeddings @ emb.T  # cosine similarity
    top_idxs = np.argsort(scores.ravel())[::-1][:top_k]
    return [st.session_state.chunks[i] for i in top_idxs]


def ask_gemini(query: str) -> str:
    context = "\n\n".join(retrieve(query))
    prompt = f"""You are an expert document assistant.

Context from PDF:
{context}

Question / Task:
{query}

Provide a clear, well-structured answer based only on the context above."""
    response = st.session_state.gemini_model.generate_content(prompt)
    return response.text


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your key from https://aistudio.google.com",
    )

    st.markdown("---")
    st.markdown("## 📄 Upload PDF")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and api_key:
        if st.button("🚀 Process PDF", use_container_width=True):
            with st.spinner("Loading embedding model…"):
                st.session_state.embedding_model = load_embedding_model()

            with st.spinner("Extracting text…"):
                text = extract_text(uploaded_file)

            with st.spinner("Chunking & indexing…"):
                chunks = chunk_text(text)
                embeddings = build_index(chunks, st.session_state.embedding_model)
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings

            genai.configure(api_key=api_key)
            st.session_state.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chat_history = []
            st.success(f"✅ Indexed {len(chunks)} chunks from **{uploaded_file.name}**")

    st.markdown("---")

    # Status
    ready = st.session_state.embeddings is not None
    badge = '<span class="status-badge badge-ready">● Ready</span>' if ready else '<span class="status-badge badge-waiting">○ Awaiting PDF</span>'
    st.markdown(f"**Status:** {badge}", unsafe_allow_html=True)

    if ready:
        st.markdown(f"**File:** {st.session_state.pdf_name}")
        st.markdown(f"**Chunks:** {len(st.session_state.chunks)}")

    if st.session_state.chat_history and st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='opacity:.5'>RAG · sentence-transformers · FAISS · Gemini</small>",
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 📄 RAG PDF Assistant")
st.markdown("Upload a PDF, then ask anything about it.")

if not api_key:
    st.info("👈 Enter your **Gemini API key** in the sidebar to get started.")
elif st.session_state.embeddings is None:
    st.info("👈 Upload a PDF and click **Process PDF** to begin.")
else:
    # Quick-action buttons
    st.markdown("#### Quick actions")
    cols = st.columns(3)
    quick = {
        "📋 Summarize": "Summarize the entire PDF in detail with key points and conclusions.",
        "🔑 Key Points": "What are the most important key points from this document?",
        "❓ Main Topic": "What is the main topic or purpose of this document?",
    }
    for col, (label, prompt) in zip(cols, quick.items()):
        if col.button(label, use_container_width=True):
            with st.spinner("Thinking…"):
                answer = ask_gemini(prompt)
            st.session_state.chat_history.append(("user", prompt))
            st.session_state.chat_history.append(("ai", answer))
            st.rerun()

    st.markdown("---")

    # Chat history
    if st.session_state.chat_history:
        st.markdown("#### Conversation")
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(
                    f'<div class="clearfix"><div class="chat-bubble-user">🧑 {msg}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="clearfix"><div class="chat-bubble-ai">🤖 {msg}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='clear:both'></div>", unsafe_allow_html=True)

    # Input
    st.markdown("#### Ask a question")
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your question",
            placeholder="What does this document say about…?",
            label_visibility="collapsed",
            height=80,
        )
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

    if submitted and user_input.strip():
        with st.spinner("Searching & generating…"):
            answer = ask_gemini(user_input.strip())
        st.session_state.chat_history.append(("user", user_input.strip()))
        st.session_state.chat_history.append(("ai", answer))

        # Download last answer
        st.download_button(
            "⬇️ Download last answer",
            data=answer,
            file_name="rag_answer.txt",
            mime="text/plain",
        )
        st.rerun()
