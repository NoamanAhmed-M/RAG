
import streamlit as st
import sys
import os
from Query import query_rag
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="RAG Chat Interface",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 70%;
        left: 15%;
    }
    .stChatMessage {
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "k_value" not in st.session_state:
    st.session_state.k_value = 5

# Title and description
st.title(" RAG Document Chat")
st.markdown("Ask questions about your indexed documents using Retrieval-Augmented Generation")

# Sidebar
with st.sidebar:
    st.header(" Configuration")
    
    # Settings
    st.session_state.k_value = st.slider(
        "Documents to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of most relevant documents to use"
    )
    
    # Clear chat button
    if st.button(" Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Info section
    st.markdown("###  About")
    st.markdown("""
    This interface uses:
    - **ChromaDB** for vector storage
    - **Ollama** with Mistral model
    - **LangChain** for RAG pipeline
    
    Documents are retrieved based on semantic similarity.
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f" Sources ({len(message['sources'])})"):
                for i, source in enumerate(message["sources"]):
                    if source:
                        st.markdown(f"**Document {i+1}:** `{source}`")
                    else:
                        st.markdown(f"**Document {i+1}:** Unknown source")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner(" Searching documents..."):
            try:
                # Call your existing query_rag function
                # Note: We need to modify query_rag to return sources
                response = query_rag(prompt)
                
                # Since query_rag prints and returns response, we need to capture it
                # Let's create a wrapper that captures the output
                import io
                from contextlib import redirect_stdout
                
                # Create string buffer to capture output
                output_buffer = io.StringIO()
                
                with redirect_stdout(output_buffer):
                    response_text = query_rag(prompt)
                
                # Parse the output (this depends on your query_rag output format)
                output_text = output_buffer.getvalue()
                
                # Display response
                st.markdown(response_text)
                
                # Try to extract sources from the printed output
                sources = []
                if "Sources:" in output_text:
                    sources_section = output_text.split("Sources:")[1].strip()
                    sources = eval(sources_section) if sources_section else []
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.divider()
st.caption("Built with Python, LangChain, ChromaDB, and Ollama")