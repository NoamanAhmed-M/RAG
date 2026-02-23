from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from pydantic import BaseModel, Field
from gen_ans import format_docs
from langchain_ollama import ChatOllama

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in 'generation' answer."""

    binary_score: str = Field(
        ...,
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# LLM 
model = ChatOllama(model="mistral")
structured_llm_grader = model.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation>"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
def hallucination_check (docs_to_use, generation):
    response = hallucination_grader.invoke({"documents": format_docs(docs_to_use), "generation": generation})
    return(response)