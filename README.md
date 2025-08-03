# Modular PDF Chatbot RAG with FastAPI, ChromaDB & Streamlit
## Introduction
This project is a modular **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and chat with an AI assistant that answers queries based on the document content. It features a microservice architecture with a decoupled **FastAPI backend** and **Streamlit frontend**, using **ChromaDB** as the vector store and **OpenAI GPT-3.5-Turbo** as the LLM engine.

---

## Project Structure

```
PDF_chatbot_RAG/
├── client/         # Streamlit Frontend
│   |──components/
|   |  |──chatUI.py
|   |  |──history_download.py
|   |  |──upload.py
|   |──utils/
|   |  |──api.py
|   |──app.py
|   |──config.py
|   |──requirements.txt
├── server/         # FastAPI Backend
│   ├──chroma_db/ ....after run
|   |──modules/
│      ├── load_vectorestore.py
│      ├── pdf_handler.py
|   |──uploaded_pdfs/ ....after run
│   ├── logger.py
│   └── main.py
|   |──requirements.txt
└── README.md
```

---

## Features

- Upload and parse PDFs
- Embed document chunks with OpenAI embeddings
- Store embeddings in ChromaDB
- Query documents using OpenAI gpt-3.5-turbo model
- Microservice architecture (Streamlit client + FastAPI server)
- Support Memory feature for keeping dialog contexts

---

## How to use

### 1. Clone the Repository

```bash
git clone https://github.com/yonglu312/pdf_chatbot_rag.git
cd pdf_chatbot_rag
```

### 2. Setup the Backend (FastAPI)

```bash
cd server
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set your OpenAI API Key (.env)
OPENAI_API_KEY="your_key_here"

# Run the FastAPI server
uvicorn main:app --reload
```

### 3. Setup the Frontend (Streamlit)

```bash
cd ../client
pip install -r requirements.txt  # if you use a separate venv for client
streamlit run app.py
```

---

## API Endpoints (FastAPI)

- `POST /upload_pdfs/` — Upload PDFs and build vectorstore
- `POST /ask/` — Send a query and receive answers

Testable via Postman or directly from the Streamlit frontend.

---

## Credits

- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- [Snsupratim](https://github.com/snsupratim/RagBot-2.0)
---
