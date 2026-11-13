
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Educational data sources

SOURCE_URLS = [
    "https://developers.google.com/machine-learning/crash-course/linear-regression",
    "https://developers.google.com/machine-learning/crash-course/logistic-regression",
    "https://developers.google.com/machine-learning/crash-course/neural-networks",
]

# Step 2: Load documents

print("Loading educational documents...")

loader = WebBaseLoader(
    web_paths=SOURCE_URLS,
    requests_kwargs={
        "headers": {"User-Agent": "Mozilla/5.0"}
    }
)

documents = loader.load()

documents = [
    doc for doc in documents
    if doc.page_content.strip()
]

if not documents:
    raise RuntimeError("No documents were loaded. Check the source URLs.")

print(f"Loaded {len(documents)} documents.")

# Step 3: Split documents into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

# Step 4: Load HuggingFace embeddings

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 5: Create FAISS vector database

print("Creating FAISS vector database...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# Step 6: Save vector database

vectorstore.save_local("vectorstore")

print("Vector database created successfully!")

print(f"Total documents: {len(documents)}")
print(f"Total chunks: {len(chunks)}")
print("Saved inside vectorstore folder.")
