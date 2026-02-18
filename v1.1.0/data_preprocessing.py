from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import shutil
from embedding_functions import get_embedding_function
from langchain_community.vectorstores import Chroma
import ollama
from PIL import Image
from pathlib import Path
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DATA_PATH = r"Data"
CHROMA_PATH = r"chroma_db"

def main():

    # Create (or update) the data store.
    documents = load_documents(DATA_PATH)
    chunks = split_document(documents)
    add_to_chroma(chunks)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
PDF_EXTENSIONS = {".pdf"}

VISION_MODEL = "qwen2.5vl"   

def ollama_ocr_image(image_path):
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Extract all readable text from this image. Return only raw text.",
                "images": [image_path],
            }
        ],
    )
    return response["message"]["content"]

#load Data
def load_documents(data_path):
    documents = []

    for file_path in Path(data_path).glob("*"):
        ext = file_path.suffix.lower()

        # ---- PDF Handling ----
        if ext in PDF_EXTENSIONS:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()

            # If empty → scanned PDF → convert to images → Ollama OCR
            if not any(doc.page_content.strip() for doc in docs):
                print(f"OCR fallback for scanned PDF: {file_path.name}")
                images = convert_from_path(str(file_path))

                for i, img in enumerate(images):
                    temp_path = f"temp_page_{i}.png"
                    img.save(temp_path)

                    text = ollama_ocr_image(temp_path)

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(file_path),
                                "page": i
                            }
                        )
                    )

                    os.remove(temp_path)
            else:
                documents.extend(docs)

        # ---- Image Handling ----
        elif ext in IMAGE_EXTENSIONS:
            print(f"OCR for image file: {file_path.name}")
            text = ollama_ocr_image(str(file_path))

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "page": 0
                    }
                )
            )

        else:
            print(f"Skipping unsupported file: {file_path.name}")

    return documents

#split data
def split_document(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
#save data 2 chromadb

import numpy as np
from typing import List
def add_to_chroma(chunks: list[Document], similarity_threshold=0.999, batch_size=100):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Get existing items
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Step 1: Filter by ID - remove chunks with existing IDs
    id_filtered_chunks = [
        chunk for chunk in chunks_with_ids 
        if chunk.metadata["id"] not in existing_ids
    ]
    
    print(f"After ID filtering: {len(id_filtered_chunks)} chunks remain")
    
    if not id_filtered_chunks:
        print("No new documents to add (all IDs already exist)")
        return

    # Step 2: Check similarity for remaining chunks
    new_chunks = []
    embedding_function = get_embedding_function()
    
    # Get the underlying collection for direct querying
    collection = db._collection
    
    # Process in batches to save RAM
    for i in range(0, len(id_filtered_chunks), batch_size):
        batch = id_filtered_chunks[i:i + batch_size]
        batch_texts = [chunk.page_content for chunk in batch]
        
        # Generate embeddings for the batch
        batch_embeddings = embedding_function.embed_documents(batch_texts)
        
        for chunk, embedding in zip(batch, batch_embeddings):
            # Query for the most similar existing document using the collection
            query_result = collection.query(
                query_embeddings=[embedding],
                n_results=1
            )
            
            # Check if we found any similar documents
            if query_result['distances'] and len(query_result['distances'][0]) > 0:
                distance = query_result['distances'][0][0]
                # Chroma uses L2 distance, convert to similarity
                # For normalized vectors: similarity = 1 - (distance^2 / 2)
                similarity = 1 - (distance ** 2 / 2)
                
                if similarity >= similarity_threshold:
                    print(f"Similar document found (similarity: {similarity:.4f}) — skipping: {chunk.metadata['id']}")
                else:
                    # Not similar enough, add it
                    new_chunks.append(chunk)
            else:
                # No existing documents in DB, add this chunk
                new_chunks.append(chunk)

    # Step 3: Add the truly new chunks
    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add (all were too similar to existing ones)")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()