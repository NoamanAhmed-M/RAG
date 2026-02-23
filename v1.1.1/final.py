from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, model_validator
from gen_ans import format_docs
from langchain_ollama import ChatOllama

class HighlightDocuments(BaseModel):
    """Return the specific part of a document used for answering the question."""

    id: List[str] = Field(
        ...,
        description="List of id of docs used to answer the question"
    )

    source: List[str] = Field(
        ...,
        description="List of sources used to answer the question"
    )

    segment: List[str] = Field(
        ...,
        description="List of direct segments from used documents that answer the question"
    )

    @model_validator(mode='before')
    @classmethod
    def normalize_keys(cls, values):
        """Lowercase all keys to handle model returning Id, Source, Segment etc."""
        return {k.lower(): v for k, v in values.items()}

# LLM
model = ChatOllama(model="mistral")

# Parser
parser = PydanticOutputParser(pydantic_object=HighlightDocuments)

# Prompt
system = """You are an advanced assistant for document search and retrieval. You are provided with the following:
1. A question.
2. A generated answer based on the question.
3. A set of documents that were referenced in generating the answer.

Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to 
generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text 
in the provided documents.

Ensure that:
- (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
- The relevance of each segment to the generated answer is clear and directly supports the answer provided.
- (Important) If you didn't use the specific document don't mention it.

Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>

<format_instruction>
{format_instructions}
</format_instruction>
"""

prompt = PromptTemplate(
    template=system,
    input_variables=["documents", "question", "generation"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

doc_lookup = prompt | model | parser

def final(docs_to_use, question, generation):
    lookup_response = doc_lookup.invoke({
        "documents": format_docs(docs_to_use),
        "question": question,
        "generation": generation
    })
    return lookup_response