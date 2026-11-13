
import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


# -----------------------------------
# STEP 1: LOAD ENVIRONMENT VARIABLES
# -----------------------------------

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Please configure your .env file.")
    st.stop()


# -----------------------------------
# STEP 2: STREAMLIT PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 StudyMate AI")
st.caption("Your AI-powered educational learning assistant.")

st.markdown("""
Ask questions about Artificial Intelligence,
Machine Learning, Neural Networks and more.
I will answer using the educational documents
in my knowledge base.
""")


# -----------------------------------
# STEP 3: LOAD RAG COMPONENTS
# -----------------------------------

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


# -----------------------------------
# STEP 4: INITIALIZE COMPONENTS
# -----------------------------------

if not os.path.exists("vectorstore/index.faiss"):
    st.error(
        "Vector database not found. "
        "Run python create_vectorstore.py first."
    )
    st.stop()

try:
    retriever, llm = load_components()
except Exception as e:
    st.error(f"Unable to initialize chatbot: {e}")
    st.stop()


# -----------------------------------
# STEP 5: CREATE PROMPT TEMPLATE
# -----------------------------------

prompt_template = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are StudyMate AI, a helpful educational assistant.

        Your job is to explain educational concepts clearly
        and help students understand their study materials.

        Follow these rules:

        1. Answer using only the provided context.
        2. Explain concepts in simple language.
        3. Give examples whenever useful.
        4. Structure answers using headings and bullet points.
        5. If the answer is unavailable in the context, say:
           "I don't know based on the provided educational documents."
        6. Never invent facts or pretend unsupported information
           came from the documents.

        Context:
        {context}
        """
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "{question}")

])

# -----------------------------------
# STEP 6: CREATE LANGCHAIN CHAIN
# -----------------------------------

rag_chain = (
    prompt_template
    | llm
    | StrOutputParser()
)


# -----------------------------------
# STEP 7: INITIALIZE CHAT HISTORY
# -----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------------
# STEP 8: SIDEBAR
# -----------------------------------

with st.sidebar:

    st.header("🎓 StudyMate AI")

    st.write("Educational Knowledge Assistant")

    st.divider()

    st.subheader("Knowledge Topics")

    st.write("""
    - Linear Regression
    - Logistic Regression
    - Neural Networks
    """)

    st.divider()

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------
# STEP 9: DISPLAY CHAT HISTORY
# -----------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 View Sources"):

                for source in message["sources"]:
                    st.markdown(f"- [{source}]({source})")


# -----------------------------------
# STEP 10: USER INPUT
# -----------------------------------

question = st.chat_input(
    "Ask me anything about your study materials..."
)


if question:

    # Display user question

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate response

    with st.chat_message("assistant"):

        with st.spinner("Searching educational documents..."):

            try:

                # Retrieve relevant documents

                documents = retriever.invoke(question)

                # Prepare context

                context = "\n\n".join([
                    f"Source: {doc.metadata.get('source', 'Educational document')}\n"
                    f"{doc.page_content}"
                    for doc in documents
                ])

                # Prepare conversation history

                chat_history = []

                for message in st.session_state.messages[:-1][-6:]:

                    if message["role"] == "user":
                        chat_history.append(
                            HumanMessage(content=message["content"])
                        )

                    elif message["role"] == "assistant":
                        chat_history.append(
                            AIMessage(content=message["content"])
                        )

                # Generate final answer

                response = rag_chain.invoke({

                    "context": context,
                    "question": question,
                    "chat_history": chat_history

                })

                st.markdown(response)

                # Collect source references

                sources = list(dict.fromkeys(
                    doc.metadata.get("source")
                    for doc in documents
                    if doc.metadata.get("source")
                ))

                with st.expander("📚 View Sources"):

                    for source in sources:
                        st.markdown(f"- [{source}]({source})")

                # Save assistant response

                st.session_state.messages.append({

                    "role": "assistant",
                    "content": response,
                    "sources": sources

                })

            except Exception as e:

                st.error(f"An error occurred: {e}")
