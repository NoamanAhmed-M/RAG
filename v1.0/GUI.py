
import streamlit as st
import sys
import os
from Query import query_rag
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="RAG Chat Interface",
    page_icon="",
    layout="wide"
)

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

if "messages" not in st.session_state:
    st.session_state.messages = []

if "k_value" not in st.session_state:
    st.session_state.k_value = 5

st.title(" RAG Document Chat")
st.markdown("Ask questions about your indexed documents using Retrieval-Augmented Generation")

with st.sidebar:
    st.header(" Configuration")
    
    st.session_state.k_value = st.slider(
        "Documents to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of most relevant documents to use"
    )
    
    if st.button(" Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.markdown("###  About")
    st.markdown("""
    This interface uses:
    - **ChromaDB** for vector storage
    - **Ollama** with Mistral model
    - **LangChain** for RAG pipeline
    
    Documents are retrieved based on semantic similarity.
    """)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f" Sources ({len(message['sources'])})"):
                for i, source in enumerate(message["sources"]):
                    if source:
                        st.markdown(f"**Document {i+1}:** `{source}`")
                    else:
                        st.markdown(f"**Document {i+1}:** Unknown source")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner(" Searching documents..."):
            try:
                response = query_rag(prompt)

                import io
                from contextlib import redirect_stdout
                
                output_buffer = io.StringIO()
                
                with redirect_stdout(output_buffer):
                    response_text = query_rag(prompt)
                
                output_text = output_buffer.getvalue()
                
                st.markdown(response_text)
                
                sources = []
                if "Sources:" in output_text:
                    sources_section = output_text.split("Sources:")[1].strip()
                    sources = eval(sources_section) if sources_section else []
                
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
