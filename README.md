
# 🎓 StudyMate AI

### Your AI-Powered Educational Learning Assistant

StudyMate AI is a Retrieval-Augmented Generation (RAG)
chatbot designed to help students understand Artificial
Intelligence and Machine Learning concepts.

## 🚀 Features

- AI-powered educational chatbot
- Document-based question answering
- Semantic search using HuggingFace embeddings
- FAISS vector database
- Groq LLM integration
- Streamlit chat interface
- Source document references
- Conversation history

## 🛠️ Technology Stack

- Python
- LangChain
- HuggingFace Embeddings
- FAISS
- Groq
- Streamlit

## 🧠 RAG Architecture

Document Loading
      ↓
Text Splitting
      ↓
HuggingFace Embeddings
      ↓
FAISS Vector Database
      ↓
Retriever
      ↓
Groq LLM
      ↓
Final Answer

## 📂 Project Structure

StudyMate-AI/
│
├── create_vectorstore.py
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

## ⚙️ Installation

1. Clone the repository.

git clone YOUR_REPOSITORY_URL

2. Navigate into the project.

cd StudyMate-AI

3. Create a virtual environment.

python -m venv .venv

4. Activate the environment.

Windows:

.venv\Scripts\activate

5. Install dependencies.

pip install -r requirements.txt

6. Configure your Groq API key.

Create a .env file:

GROQ_API_KEY=your_api_key

7. Create the vector database.

python create_vectorstore.py

8. Run the application.

streamlit run app.py

## 📚 Educational Data Sources

Google Machine Learning Crash Course:

- Linear Regression
- Logistic Regression
- Neural Networks

Source:
https://developers.google.com/machine-learning/crash-course

The referenced educational content is provided under
Creative Commons Attribution 4.0, except where otherwise
noted. Please retain attribution to the original authors.

## 👨‍💻 Author

Eswar Sai Krishna

## 📌 Project Domain

Education – Artificial Intelligence and Machine Learning
