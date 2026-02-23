import argparse
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import sys
import os
import traceback
from embedding_functions import get_embedding_function
from data_real import relevancy_check
from gen_ans import generate_res
from halluciation_check import hallucination_check
from final import final
from langchain_ollama import ChatOllama

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    from data_real import relevancy_check
    from gen_ans import generate_res
    from halluciation_check import hallucination_check
    from final import final

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    print("searching database")
    results = db.similarity_search_with_score(query_text, k=5)

    print("searching database done")
    print("relevancy_check")
    docs_to_use = relevancy_check(query_text, results)
    print("relevancy_check done")

    #Interrupt if no relevant documents were found 
    if not docs_to_use:
        print("no relevant documents found, generating model response")
        no_data_prompt = f"""You are a knowledgeable assistant. 
The user asked: "{query_text}"

You searched your knowledge base thoroughly but found no relevant information related to this query.
Respond naturally and honestly: let the user know that your knowledge base does not contain 
information relevant to their question. Be polite, concise, and suggest they try rephrasing 
or ask a different question."""

        model = ChatOllama(model="mistral")
        no_data_response = model.invoke(no_data_prompt)
        print("no data response generated")
        return no_data_response, no_data_response, "not_applicable"
    
    for doc in docs_to_use:
        print("Metadata keys:", doc.metadata)
    print("generating response")
    response_llm = generate_res(docs_to_use, query_text)
    print("generating done")

    print("hallucination checking")
    hallucination_binary_score = hallucination_check(docs_to_use, response_llm)
    print("hallucination checking done")

    print("generating final answer")
    final_answer = final(docs_to_use, query_text, response_llm)
    print("generating final answer done")

    return response_llm, final_answer, hallucination_binary_score

if __name__ == "__main__":
    main()