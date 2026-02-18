from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_function():
    return OllamaEmbeddings(
        model="mistral:7b",
        base_url="http://localhost:11434"
    )
