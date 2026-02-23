from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_community.llms.ollama import Ollama
from langchain_ollama import ChatOllama
# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


model = ChatOllama(model="mistral")
structured_llm_grader = model.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
def relevancy_check(question, docs):
    docs_to_use = []
    for doc, score in docs: 
        print(doc.page_content, '\n', '-'*50)
        res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        print(res, '\n')
        if res.binary_score == 'yes':
            docs_to_use.append(doc)
    return docs_to_use