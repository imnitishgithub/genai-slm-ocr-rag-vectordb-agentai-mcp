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