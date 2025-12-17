Building this "stack" from scratch on a local Windows laptop is the absolute best way to demystify GenAI. By running it locally with minimal resources, you learn exactly where the bottlenecks are (latency, RAM, disk I/O) and how the components interact without relying on paid APIs.

We will build a Local GenAI Lab using a "Small Language Model" (SLM) to keep it fast on your CPU.

The Architecture

Input: Images/PDFs (OCR).
Processing: Text Splitting (Chunking) & Embedding.
Storage: Local Vector Database (ChromaDB).
Intelligence: Quantized LLM (via llama.cpp).
Agent: A reasoning loop that decides whether to use the VectorDB or answer directly.
Protocol: MCP (Model Context Protocol) to standardize how we expose tools.
UI: Streamlit (Web Interface).

Prerequisites (Windows)

Before we write code, we need the foundational tools.
Install Python (3.10 or 3.11): Download from python.org. Ensure you check "Add Python to PATH".

https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe.

<img width="688" height="419" alt="image" src="https://github.com/user-attachments/assets/85c56a12-d495-4558-8b54-240ea534c4c3" />

 
Install Visual Studio Build Tools: (Crucial for llama-cpp-python on Windows).
Download "Visual Studio Build Tools".
Select "Desktop development with C++" workload during installation.

Install Tesseract OCR:
Download the Windows installer (e.g., from UB-Mannheim's GitHub).
Install it (e.g., to C:\Program Files\Tesseract-OCR).
Important: Add this path to your System Environment Variables PATH.


 <img width="975" height="390" alt="image" src="https://github.com/user-attachments/assets/c356e13b-67ce-482e-a874-7765911b7b62" />


 <img width="975" height="221" alt="image" src="https://github.com/user-attachments/assets/c5cf8897-cea7-47d5-890a-7919bc104938" />


Create a Project Folder: Open PowerShell:
mkdir GenAI_Lab
cd GenAI_Lab
py -3.11 -m venv venv
.\venv\Scripts\activate
python –version
If it says Python 3.11.x, you are safe.
 

Step 1: Installation & Quantization (The Engine)

Concept: Quantization reduces the precision of model weights (e.g., from 16-bit float to 4-bit integer). This drops memory usage from ~16GB to ~4GB, allowing it to run on your laptop RAM.
We won't quantize a model (which requires massive RAM); we will use a pre-quantized model in the GGUF format.

Install Libraries:


Correct Command which worked in demo:

pip install langchain==0.3.12 langchain-community langchain-huggingface chromadb pytesseract pypdf sentence-transformers llama-cpp-python streamlit mcp easyocr

(Note: If llama-cpp-python fails, ensure VS Build Tools are installed properly).

 <img width="741" height="714" alt="image" src="https://github.com/user-attachments/assets/1b6b55b3-5e0e-48df-9632-4f119674ff66" />

 <img width="975" height="520" alt="image" src="https://github.com/user-attachments/assets/3efe3a8f-de9b-45a2-8bbb-ab1c90bdc001" />

Download a Model: Create a folder named models. Download Llama-3.2-3B-Instruct-Q4_K_M.gguf (approx 2GB) from HuggingFace (search for "bartowski/Llama-3.2-3B-Instruct-GGUF") and save it inside models/.

https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
 
 <img width="1224" height="855" alt="image" src="https://github.com/user-attachments/assets/9e69ce11-3946-4f00-b46b-d64cacc80993" />

 
Step 2: The Core Script (OCR, RAG, VectorDB)

Create a file named backend.py. This handles ingestion, embedding, and retrieval.

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

Step 3: Agentic AI & MCP
Concept:

Agentic AI: The LLM isn't just a chatbot; it has "tools". It decides when to use a tool (like searching the database) vs answering from memory.

MCP (Model Context Protocol): A standard way to define these tools so different AI clients can use them universally.



Create a file named agent.py.

from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from backend import get_retriever
from mcp.server.fastmcp import FastMCP

# 1. Load the Quantized Model (GGUF)
# n_ctx=2048 is the context window. n_gpu_layers=0 means run entirely on CPU.
llm = LlamaCpp(
    model_path="./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    temperature=0.1,
    max_tokens=512,
    n_ctx=2048,
    verbose=False
)

# 2. Define the RAG Tool
def query_knowledge_base(query):
    """Useful for answering questions about the uploaded documents."""
    retriever = get_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain.run(query)

# 3. Create the Agent
# The agent has access to the "Knowledge Base" tool.
tools = [
    Tool(
        name="Knowledge Base",
        func=query_knowledge_base,
        description="Use this tool when answering questions about user documents, PDF, or OCR data."
    )
]

# ZERO_SHOT_REACT_DESCRIPTION: The agent uses "Reasoning" + "Acting" logic.
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    handle_parsing_errors=True
)

# 4. MCP Implementation (Mini Server)
# This creates a lightweight server that exposes our RAG capability via MCP protocol
mcp_server = FastMCP("LocalGenAI_Agent")

@mcp_server.tool()
def ask_document(question: str) -> str:
    """Ask a question to the local document store via RAG"""
    return query_knowledge_base(question)

def run_agent(user_input):
    return agent.run(user_input)

Step 4: The GUI (Streamlit)

Create a file named app.py. This ties everything together visually.
import streamlit as st
import os
from backend import ingest_document
from agent import run_agent

st.set_page_config(page_title="Local GenAI Lab", layout="wide")

st.title("🤖 Local GenAI Workbench")
st.markdown("### Stack: OCR -> Chunking -> Embeddings -> VectorDB -> Quantized LLM -> Agent")

# Sidebar for Ingestion
with st.sidebar:
    st.header("1. Ingestion (RAG)")
    uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg"])
    
    if uploaded_file:
        # Save file locally temporarily
        save_path = os.path.join("temp_data", uploaded_file.name)
        os.makedirs("temp_data", exist_ok=True)
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        file_type = "image" if uploaded_file.name.endswith(("png", "jpg")) else "pdf"
        
        if st.button("Process & Embed"):
            with st.spinner("OCR processing, Chunking & Embedding..."):
                ingest_document(save_path, file_type=file_type)
            st.success("✅ Added to Knowledge Base!")

# Main Chat Interface
st.header("2. Agentic Chat")
user_query = st.text_input("Ask something about your docs or general knowledge:")

if st.button("Ask Agent"):
    if user_query:
        with st.spinner("Agent is Thinking (Reasoning)..."):
            # The agent will decide if it needs to read the docs or just chat
            response = run_agent(user_query)
            st.markdown(f"**Answer:** {response}")
            
            # Visualizing the components
            st.info("💡 **Learning Note:** If the Agent used the 'Knowledge Base' tool in the logs, it performed RAG. If not, it used its internal weights.")


<img width="975" height="265" alt="image" src="https://github.com/user-attachments/assets/12a37681-b255-423f-ac75-8fd8845a853f" />


Step 5: How to Run It

Start the App:

streamlit run app.py
 

<img width="975" height="573" alt="image" src="https://github.com/user-attachments/assets/84e625fe-287e-4163-99bf-a426c6c1d876" />




<img width="975" height="435" alt="image" src="https://github.com/user-attachments/assets/e549c7d4-6a30-4b43-8e80-ffd86ec1940d" />

 

The Fix: Remove the Bad Package & Reinstall the Official One

You need to remove the confusing package and reinstall the real one. Please run these commands one by one in your terminal (where you see (venv)):
1. Uninstall the conflicting packages:
pip uninstall -y langchain langchain-community langchain-core langchain-classic

2. Reinstall the official, correct versions:
pip install langchain==0.3.12 langchain-community langchain-huggingface chromadb pytesseract pypdf sentence-transformers llama-cpp-python streamlit mcp easyocr
 

pip show langchain
 
<img width="975" height="155" alt="image" src="https://github.com/user-attachments/assets/0507ad59-17b4-4d0c-b8f4-4581a1c15afb" />

<img width="975" height="116" alt="image" src="https://github.com/user-attachments/assets/7c008550-cedf-4022-895c-c4365cdaea8d" />

You can safely ignore this error for now.

Here is the simple explanation of what is happening:
1. It is a "False Alarm" for your specific needs
The error is complaining about a side-tool called langgraph-prebuilt. This tool is asking for langchain-core version 1.0+, which is a weird version requirement that conflicts with the standard langchain library you just installed.
However, your code does not seem to use langgraph-prebuilt. Your code uses langchain.chains and langchain_community, both of which are now successfully installed.

streamlit run app.py

<img width="975" height="549" alt="image" src="https://github.com/user-attachments/assets/6028e3ff-611a-44c5-9c42-7280108d75eb" />


 <img width="975" height="183" alt="image" src="https://github.com/user-attachments/assets/a51f3b7f-ec6c-467d-b337-88b2bd370c7a" />

The Workflow (Learning Path):

Upload an Image: Upload a screenshot of text. Watch the terminal. You will see Tesseract OCR extract the text.
Click Process: Watch the terminal. You will see Chunking (splitting text) and Embedding (creating chroma_db folder).
 
 <img width="342" height="705" alt="image" src="https://github.com/user-attachments/assets/1b16cbd1-c710-44f2-b9be-056580987553" />

<img width="975" height="543" alt="image" src="https://github.com/user-attachments/assets/256a999b-2a54-4a3f-aed9-6708bee2c2f5" />

 <img width="975" height="546" alt="image" src="https://github.com/user-attachments/assets/ad749eb0-bed8-4195-a674-6b10f1b005cb" />

<img width="975" height="546" alt="image" src="https://github.com/user-attachments/assets/e58be036-bc28-459d-be04-12a5eabecc06" />

<img width="975" height="546" alt="image" src="https://github.com/user-attachments/assets/35e71c5a-cf96-4a0d-8a72-389a15a3bfb2" />

<img width="975" height="544" alt="image" src="https://github.com/user-attachments/assets/eace7871-1e9b-4e74-aa1a-f06ffeead56e" />

<img width="975" height="410" alt="image" src="https://github.com/user-attachments/assets/5fd477c2-e69e-44fc-9dc7-8723cb9e7032" />

<img width="975" height="522" alt="image" src="https://github.com/user-attachments/assets/e3026a2c-1452-4cfd-a88e-520fe204efe8" />

<img width="975" height="944" alt="image" src="https://github.com/user-attachments/assets/0cd57794-2d82-4666-8808-08ce81afd4e9" />

This screenshot confirms that your installation is 100% fixed and the application is running! However, now you are seeing a logic issue within the AI agent itself.
Here is the breakdown of exactly what is happening in the logs:
1. The Good News (Top Section)
•	OCR: Reading ...: The app successfully used easyocr to read the file you uploaded (Agent-to-Agent-Protocol-Screenshot-File.png).
•	Data stored in VectorDB: It successfully chunked the text, created embeddings using sentence-transformers, and stored them in chromadb.
•	Status: The "Backend" is working perfectly. All those libraries you struggled to install are doing their job.
2. The Bad News (The Loop)
The Agent enters a "loop of confusion" where it tries to answer your question but fails to use the search tool correctly.
•	The Goal: You asked: "I need to look up the differences between MCP and A2A."
•	The Mistake:
o	The Agent knows it needs to use the tool named Knowledge Base.
o	However, instead of just typing Knowledge Base, it types Use Knowledge Base (adding the word "Use").
o	The Error Message: Observation: Use Knowledge Base is not a valid tool, try one of [Knowledge Base].
•	The Loop:
o	The Agent sees the error.
o	It thinks, "Oh, I need to use the Knowledge Base tool."
o	It tries again: Action: Use Knowledge Base.
o	The system rejects it again.
o	This repeats until the agent gives up (Finished chain).
Why is this happening?
This is a common "Prompt Engineering" issue with smaller models (like Llama 2 or older versions). The AI model is being too conversational.
•	System expects: Action: Knowledge Base
•	AI outputs: Action: Use Knowledge Base
How to Fix This
You need to make the AI smarter about how it calls the tool. You have two options:
Option 1: Rename the Tool (Easiest) In your code (likely backend.py or where the tool is defined), change the tool's name to match what the AI wants to say.
•	Change tool name from: "Knowledge Base"
•	To: "Use Knowledge Base"
Option 2: Fix the Prompt (Better) In agent.py, where you initialize the agent, you can add a stronger instruction to the system prompt:
"When using a tool, output ONLY the exact tool name in the Action field. Do not add the word 'Use'."
Summary: You have solved the installation nightmare! Now you are just debugging the AI's behavior, which is the fun part of development.




Ask a Question: Ask "What does the document say about X?".
Watch the Agent: In your terminal, you will see the Agent's "Thought Process" (The ReAct pattern):
Thought: I need to check the document.
Action: Knowledge Base.
Observation: [Retrieved text].
Final Answer: [Summary].



Concept	Implementation in this Project
OCR	pytesseract reads text from .png/.jpg.
Chunking	RecursiveCharacterTextSplitter breaks text into 500-char blocks.
Embedding	sentence-transformers converts blocks to vector numbers.
VectorDB	ChromaDB stores these numbers on your hard drive.
Quantization	llama.cpp loads the .gguf model using 4-bit integers (Low RAM).
RAG	The query_knowledge_base function retrieves relevant chunks before answering.
Agentic AI	The agent wrapper decides if it needs to use the RAG function or not.
MCP	The mcp snippet shows how you would standardise this tool for other AI clients.

