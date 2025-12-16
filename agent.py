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