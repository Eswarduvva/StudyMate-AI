

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="StudyMate AI | Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f7fc;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}

[data-testid="stSidebar"] {
    background: #101b35;
}

[data-testid="stSidebar"] * {
    color: #edf2ff !important;
}

.brand {
    font-size: 25px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 4px;
}

.brand-subtitle {
    color: #aab8d5;
    font-size: 12px;
    margin-bottom: 25px;
}

.sidebar-label {
    color: #8fa5cc !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-top: 25px;
}

.hero {
    background: linear-gradient(120deg, #14264d, #2855a6);
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    color: #dbe7ff;
    font-size: 15px;
    margin-bottom: 0;
}

.section-title {
    font-size: 19px;
    font-weight: 700;
    color: #172747;
    margin: 22px 0 12px 0;
}

.info-card {
    background: white;
    border: 1px solid #e3e9f4;
    border-radius: 15px;
    padding: 18px;
    min-height: 110px;
}

.info-card h4 {
    color: #1b376a;
    font-size: 15px;
    margin: 0 0 8px 0;
}

.info-card p {
    color: #65738b;
    font-size: 12px;
    margin: 0;
}

[data-testid="stChatMessage"] {
    background: white;
    border: 1px solid #e7ebf3;
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 12px;
}

[data-testid="stChatInput"] {
    border-radius: 15px;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid #dce5f4;
    background: white;
    color: #24477f;
    font-weight: 600;
    min-height: 48px;
    text-align: left;
}

.stButton > button:hover {
    border-color: #4679d4;
    color: #24477f;
    background: #edf4ff;
}

[data-testid="stExpander"] {
    border: 1px solid #e2e8f2;
    border-radius: 12px;
    background: white;
}

.footer {
    color: #8a96aa;
    font-size: 12px;
    text-align: center;
    margin-top: 35px;
}


/* Main page text */
.stApp,
.stApp p,
.stApp label,
.stMarkdown,
.stMarkdown p,
.stMarkdown li {
    color: #24324b;
}

/* Main headings */
h1, h2, h3, h4 {
    color: #14264d !important;
}

/* Hero banner */
.hero h1 {
    color: #ffffff !important;
}

.hero p {
    color: #e2eaff !important;
}

/* Chat messages */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #26364f !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    color: #172747 !important;
    -webkit-text-fill-color: #edf2ff !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #718096 !important;
    -webkit-text-fill-color: #718096 !important;
}

/* Sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    color: #edf2ff !important;
}

/* Source links */
.stMarkdown a {
    color: #2563eb !important;
}

/* Cards */
.info-card h4 {
    color: #1b376a !important;
}

.info-card p {
    color: #52627b !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- API KEY ----------------

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Please configure your .env file.")
    st.stop()

# ---------------- LOAD RAG COMPONENTS ----------------

if not os.path.exists("vectorstore/index.faiss"):
    st.error("Vector database not found. Run python create_vectorstore.py first.")
    st.stop()

@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    return retriever, llm

try:
    retriever, llm = load_components()
except Exception as e:
    st.error(f"Unable to initialize chatbot: {e}")
    st.stop()

# ---------------- PROMPT ----------------

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
    You are StudyMate AI, an educational learning assistant.

    Explain concepts in simple language.
    Use headings, bullet points and examples when helpful.
    Answer using only the supplied educational context.
    If the information is not available, clearly say so.
    Never invent facts or sources.

    Context:
    {context}
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

rag_chain = prompt_template | llm | StrOutputParser()

# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown('<div class="brand">🎓 StudyMate AI</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">YOUR PERSONAL AI LEARNING COMPANION</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown('<div class="sidebar-label">LEARNING TOPICS</div>',
                unsafe_allow_html=True)

    st.markdown("""
    📘 Linear Regression

    📗 Logistic Regression

    🧠 Neural Networks

    🤖 Machine Learning
    """)

    st.divider()

    st.markdown('<div class="sidebar-label">YOUR WORKSPACE</div>',
                unsafe_allow_html=True)

    st.write("💬 Ask educational questions")
    st.write("📚 Explore concepts")
    st.write("🔎 Find document sources")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown(
        '<div class="brand-subtitle">Powered by LangChain · FAISS · Groq</div>',
        unsafe_allow_html=True
    )

# ---------------- MAIN HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>Learn smarter with StudyMate AI ✨</h1>
    <p>Your AI-powered educational assistant.
    Ask questions, understand difficult concepts and explore AI/ML topics.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- WELCOME SCREEN ----------------

if not st.session_state.messages:

    st.markdown('<div class="section-title">What would you like to learn today?</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>📘 Machine Learning</h4>
            <p>Understand algorithms, training and prediction.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Explain supervised learning", use_container_width=True):
            st.session_state.pending_question = "Explain supervised learning with an example."
            st.rerun()

        st.markdown("""
        <div class="info-card">
            <h4>📈 Regression</h4>
            <p>Explore linear and logistic regression concepts.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Compare linear and logistic regression", use_container_width=True):
            st.session_state.pending_question = "What is the difference between linear regression and logistic regression?"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🧠 Neural Networks</h4>
            <p>Discover how artificial neural networks work.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Explain neural networks", use_container_width=True):
            st.session_state.pending_question = "Explain neural networks in simple terms."
            st.rerun()

        st.markdown("""
        <div class="info-card">
            <h4>🎯 Study Support</h4>
            <p>Understand important concepts with examples.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Explain machine learning with examples", use_container_width=True):
            st.session_state.pending_question = "Explain machine learning with real-life examples."
            st.rerun()

# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:
    avatar = "🧑‍🎓" if message["role"] == "user" else "🎓"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 View educational sources"):
                for source in message["sources"]:
                    st.markdown(f"- [{source}]({source})")

# ---------------- CHAT INPUT ----------------

question = st.chat_input("Ask StudyMate AI anything about your studies...")

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("📚 Searching your educational knowledge base..."):
            try:
                documents = retriever.invoke(question)

                context = "\n\n".join([
                    f"Source: {doc.metadata.get('source', 'Educational document')}\n"
                    f"{doc.page_content}"
                    for doc in documents
                ])

                if not context.strip():
                    response = (
                        "I don't know based on the provided educational documents."
                    )
                    sources = []
                else:
                    chat_history = []

                    for message in st.session_state.messages[:-1][-6:]:
                        if message["role"] == "user":
                            chat_history.append(
                                HumanMessage(content=message["content"])
                            )
                        else:
                            chat_history.append(
                                AIMessage(content=message["content"])
                            )

                    response = rag_chain.invoke({
                        "context": context,
                        "question": question,
                        "chat_history": chat_history
                    })

                    sources = list(dict.fromkeys(
                        doc.metadata.get("source")
                        for doc in documents
                        if doc.metadata.get("source")
                    ))

                st.markdown(response)

                if sources:
                    with st.expander("📚 View educational sources"):
                        for source in sources:
                            st.markdown(f"- [{source}]({source})")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.markdown("""
<div class="footer">
    StudyMate AI · Learn • Explore • Understand
</div>
""", unsafe_allow_html=True)
