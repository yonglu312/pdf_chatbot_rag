import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

UPLOAD_DIR = "./uploaded_pdfs"
if os.path.exists(UPLOAD_DIR):
    try:
        files = os.listdir(UPLOAD_DIR)
        for filename in files:
            filepath = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
                print(f"Deteleted file: {filepath}")
    except FileNotFoundError:
        print(f"Error: file not found at {UPLOAD_DIR}")
    except Exception as e:
        print(f"An error occured: {e}")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load, split, embed and upsert PDF content
def load_vectorstore(uploaded_files):
    embed_model = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="chromadb_cs229",
        embedding_function=embed_model,
        persist_directory="./chroma_db",
    )

    file_paths = []

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    docs = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        docs.extend(chunks)
        print(f"✅ Upload complete for {file_path}")
    vector_store.add_documents(documents=docs)
