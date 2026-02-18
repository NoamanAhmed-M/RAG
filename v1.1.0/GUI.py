import streamlit as st
import sys
import os
from Query import query_rag
import warnings
from pathlib import Path
import shutil
import tempfile

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="RAG Chat Interface",
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

DATA_PATH = r"Data"

# Ensure Data directory exists
os.makedirs(DATA_PATH, exist_ok=True)

# Import functions from populate_database
def process_uploaded_files(uploaded_files):
    """Process uploaded files and add them to the database"""
    from data_preprocessing import load_documents, split_document, add_to_chroma
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    saved_files = []
    
    try:
        # Save uploaded files to temp directory for processing
        for uploaded_file in uploaded_files:
            # Save to Data directory (permanent storage)
            data_file_path = os.path.join(DATA_PATH, uploaded_file.name)
            with open(data_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Save to temp directory (for processing)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            saved_files.append(uploaded_file.name)
        
        # Process documents from temp directory
        print("loading doc")
        documents = load_documents(temp_dir)
        print("splitting doc")
        # Split documents
        chunks = split_document(documents)
        print("saving doc")
        add_to_chroma(chunks)
        
        return saved_files, len(chunks)
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

st.title("RAG Document Chat")
st.markdown("Ask questions about your indexed documents using Retrieval-Augmented Generation")

with st.sidebar:
    st.header("Configuration")
    
    st.session_state.k_value = st.slider(
        "Documents to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of most relevant documents to use"
    )
    
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # File Upload Section
    st.header("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Add files to database",
        type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        accept_multiple_files=True,
        help="Upload PDF or image files to add to the knowledge base"
    )
    
    if uploaded_files:
        if st.button("Add to Database", type="primary"):
            with st.spinner("Processing and adding documents..."):
                try:
                    saved_files, chunk_count = process_uploaded_files(uploaded_files)
                    
                    st.success(f"Successfully processed {len(saved_files)} file(s)!")
                    st.info(f"Processed {chunk_count} document chunks")
                    st.info("Duplicate chunks are automatically skipped based on content")
                    st.balloons()
                    
                    # Show uploaded files
                    with st.expander("Processed Files"):
                        for filename in saved_files:
                            st.text(f"• {filename}")
                    
                except Exception as e:
                    st.error(f"Error adding documents: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    st.divider()
    
    # Database Management
    st.header("Database Management")
    
    if st.button("Reindex All Documents", type="secondary"):
        with st.spinner("Reindexing all documents in Data folder..."):
            try:
                from data_preprocessing import main
                main()
                st.success("Database reindexed successfully!")
            except Exception as e:
                st.error(f"Error reindexing: {str(e)}")
    
    # Show existing files in database
    with st.expander("View Files in Data Folder"):
        existing_files = os.listdir(DATA_PATH) if os.path.exists(DATA_PATH) else []
        if existing_files:
            st.write(f"**Total files: {len(existing_files)}**")
            for file in sorted(existing_files):
                st.text(f"• {file}")
        else:
            st.text("No files in Data folder yet")
    
    st.divider()
    
    st.markdown("About")
    st.markdown("""
    This interface uses:
    - **ChromaDB** for vector storage
    - **Ollama** with Qwen2.5VL for OCR
    - **Mistral** for responses
    - **LangChain** for RAG pipeline
    
    **Duplicate Handling:**
    - Files are checked by chunk ID (source:page:chunk_index)
    - Same content from same file won't be added twice
    - Modified files will create new chunks
    """)

# Chat History Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"Sources ({len(message['sources'])})"):
                for i, source in enumerate(message["sources"]):
                    if source:
                        st.markdown(f"**Document {i+1}:** `{source}`")
                    else:
                        st.markdown(f"**Document {i+1}:** Unknown source")

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
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