import os
import pytesseract
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
# Point this to your Tesseract EXE if it's not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
VECTOR_DB_PATH = "./chroma_db"

# 1. OCR Function: Extracts text from images
def perform_ocr(image_path):
    print(f"👀 OCR: Reading {image_path}...")
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# 2. Chunking & Ingestion: Prepares data for the DB
def ingest_document(file_path, file_type="pdf"):
    print("✂️ Chunking and Embedding...")
    
    if file_type == "image":
        text = perform_ocr(file_path)
        # Create a mock document object
        from langchain.docstore.document import Document
        docs = [Document(page_content=text, metadata={"source": file_path})]
    else:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

    # Chunking: Breaking text into manageable pieces
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Characters per chunk
        chunk_overlap=50  # Overlap to maintain context
    )
    splits = text_splitter.split_documents(docs)

    # Embedding: Converting text to numbers (Vectors)
    # We use a small, fast model suitable for CPUs
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # VectorDB: Storing the vectors
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=VECTOR_DB_PATH
    )
    print("✅ Data stored in VectorDB")
    return vectorstore

# 3. Retrieval: Searching the DB
def get_retriever():
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function)
    return vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks