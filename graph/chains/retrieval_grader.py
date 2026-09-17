# Note that every node will run chains, and this chain will be run by the grade_documents node in the arch.png.
# This is the chains that will be used in the grade_documents node in the arch.png.

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

llm = ChatOllama(temperature=0, model="qwen3:8b")

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# Now what is going to do under the hood in the with_structured_output function.
# The LLM's going to use function calling.
# And for every LLM call we make we are going to return a Pydantic object.
# And the LLM is going to return in the schema that we want(GradeDocuments).
# In other meaning, the response from the LLM will be formatted using GradeDocuments.
structured_llm_grader = llm.with_structured_output(GradeDocuments) # To use with_structured_output function you have to make sure that the LLM supports function calling.

# System prompt.
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
