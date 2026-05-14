import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d0f1a; color: #e8e8f0; }

/* sidebar */
section[data-testid="stSidebar"] { background: #13152a; border-right: 1px solid #2a2d4a; }
section[data-testid="stSidebar"] * { color: #c8cae0 !important; }
section[data-testid="stSidebar"] h2 { color: #fff !important; font-family: 'Syne', sans-serif !important; }

/* hero */
.hero { text-align: center; padding: 60px 20px 40px; }
.hero-icon { font-size: 64px; margin-bottom: 16px; }
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7c6ff7, #4fc3f7, #7c6ff7);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 16px;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #8890b0;
    max-width: 580px;
    margin: 0 auto;
    line-height: 1.7;
}

/* how-it-works cards */
.how-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 40px 0 28px;
}
.how-card {
    background: #181b2e;
    border: 1px solid #2a2d4a;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    position: relative;
}
.how-num {
    position: absolute;
    top: 12px; right: 14px;
    font-size: 0.7rem;
    font-weight: 700;
    color: #4a4d7a;
    font-family: 'Syne', sans-serif;
}
.how-icon { font-size: 32px; margin-bottom: 12px; }
.how-label { font-size: 0.88rem; color: #8890b0; line-height: 1.5; }

/* tech pills */
.tech-row { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin-bottom: 36px; }
.tech-pill {
    background: #1e2240;
    border: 1px solid #3a3d6a;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: #9fa3c8;
}

/* callouts */
.callout {
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.95rem;
    line-height: 1.6;
}
.callout a { color: #7c6ff7; }
.callout-warn { background: #2a2010; border: 1px solid #5a4a20; color: #d4b060; }
.callout-info { background: #10202a; border: 1px solid #1a4a6a; color: #60a4d4; }

/* chat bubbles */
.chat-bubble-user {
    background: linear-gradient(135deg, #7c6ff7, #5a4fd7);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 13px 18px;
    margin: 8px 0;
    max-width: 72%;
    float: right;
    clear: both;
    font-size: 0.93rem;
    line-height: 1.5;
}
.chat-bubble-ai {
    background: #181b2e;
    color: #d8daf0;
    border: 1px solid #2a2d4a;
    border-radius: 18px 18px 18px 4px;
    padding: 13px 18px;
    margin: 8px 0;
    max-width: 78%;
    float: left;
    clear: both;
    font-size: 0.93rem;
    line-height: 1.6;
}
.clearfix::after { content: ""; display: table; clear: both; }

/* status badge */
.status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.badge-ready  { background: #0f2a1a; color: #4caf7a; border: 1px solid #1a5a30; }
.badge-waiting{ background: #2a2010; color: #c4903a; border: 1px solid #5a4010; }

/* section heading */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #6068a0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 28px 0 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
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
    return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())


def chunk_text(text: str, chunk_size=1000, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - overlap
    return chunks


def build_embeddings(chunks, model):
    embs = model.encode(chunks, show_progress_bar=False).astype("float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-9)


def retrieve(query: str, top_k=5):
    emb = st.session_state.embedding_model.encode([query]).astype("float32")
    emb = emb / np.maximum(np.linalg.norm(emb), 1e-9)
    scores = st.session_state.embeddings @ emb.T
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
    return st.session_state.gemini_model.generate_content(prompt).text


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your key free at https://aistudio.google.com",
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
                embs = build_embeddings(chunks, st.session_state.embedding_model)
                st.session_state.chunks = chunks
                st.session_state.embeddings = embs
            genai.configure(api_key=api_key)
            st.session_state.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chat_history = []
            st.success(f"✅ Indexed {len(chunks)} chunks from **{uploaded_file.name}**")

    st.markdown("---")
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
    st.markdown("<small style='opacity:.4'>RAG · sentence-transformers · Gemini 2.5</small>", unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-icon">📄</div>
    <h1 class="hero-title">RAG PDF Assistant</h1>
    <p class="hero-subtitle">
        Upload any PDF and instantly chat with it — get summaries, extract key insights,
        and ask questions in plain English. Powered by <strong>Gemini AI</strong> and
        semantic similarity search.
    </p>
</div>
""", unsafe_allow_html=True)

# Show landing content only when not yet ready
if st.session_state.embeddings is None:
    st.markdown("""
    <div class="how-grid">
        <div class="how-card">
            <div class="how-num">01</div>
            <div class="how-icon">🔑</div>
            <div class="how-label">Enter your <strong>Gemini API key</strong> in the sidebar</div>
        </div>
        <div class="how-card">
            <div class="how-num">02</div>
            <div class="how-icon">📤</div>
            <div class="how-label">Upload any <strong>PDF</strong> — paper, report, contract, book</div>
        </div>
        <div class="how-card">
            <div class="how-num">03</div>
            <div class="how-icon">🚀</div>
            <div class="how-label">Click <strong>Process PDF</strong> to embed & index it</div>
        </div>
        <div class="how-card">
            <div class="how-num">04</div>
            <div class="how-icon">💬</div>
            <div class="how-label"><strong>Chat freely</strong> — summarize, query, extract info</div>
        </div>
    </div>
    <div class="tech-row">
        <span class="tech-pill">🧠 sentence-transformers</span>
        <span class="tech-pill">🔍 Cosine similarity search</span>
        <span class="tech-pill">✨ Gemini 2.5 Flash</span>
        <span class="tech-pill">📚 RAG architecture</span>
        <span class="tech-pill">🐍 Pure Python · No FAISS</span>
    </div>
    """, unsafe_allow_html=True)

    if not api_key:
        st.markdown("""
        <div class="callout callout-warn">
            👈 <strong>Step 1:</strong> Paste your Gemini API key in the sidebar to unlock the app.
            Get one free at <a href="https://aistudio.google.com" target="_blank">aistudio.google.com</a>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="callout callout-info">
            👈 <strong>Step 2:</strong> Upload a PDF and hit <strong>Process PDF</strong> — then start chatting!
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Quick actions ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Quick Actions</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    quick = {
        "📋 Full Summary": "Summarize the entire PDF in detail with key points and conclusions.",
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

    # ── Chat history ──────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown('<div class="section-head">Conversation</div>', unsafe_allow_html=True)
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
        st.markdown("<div style='clear:both;margin-bottom:24px'></div>", unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Ask a Question</div>', unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "question",
            placeholder="e.g. What are the main findings? Who are the authors? What methodology was used?",
            label_visibility="collapsed",
            height=90,
        )
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

    if submitted and user_input.strip():
        with st.spinner("Searching document & generating answer…"):
            answer = ask_gemini(user_input.strip())
        st.session_state.chat_history.append(("user", user_input.strip()))
        st.session_state.chat_history.append(("ai", answer))
        st.download_button(
            "⬇️ Download this answer",
            data=answer,
            file_name="rag_answer.txt",
            mime="text/plain",
        )
        st.rerun()
