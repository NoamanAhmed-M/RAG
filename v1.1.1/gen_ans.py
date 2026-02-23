from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# Prompt
system = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. 
Use three-to-five sentences maximum and keep the answer concise."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>"),
    ]
)

# LLM
model = ChatOllama(model="mistral")
def format_docs(docs):
    return "\n".join(
        f"<doc{i+1}>:\n"
        f"ID:{doc.metadata.get('id', 'N/A')}\n"
        f"Page:{doc.metadata.get('page', 'N/A')}\n"
        f"Source:{doc.metadata.get('source', 'N/A')}\n"
        f"Content:{doc.page_content}\n"
        f"</doc{i+1}>\n"
        for i, doc in enumerate(docs)
    )
# Chain
rag_chain = prompt | model | StrOutputParser()

def generate_res(docs_to_use,question):
    generation = rag_chain.invoke({"documents":format_docs(docs_to_use), "question": question})
    return(generation)