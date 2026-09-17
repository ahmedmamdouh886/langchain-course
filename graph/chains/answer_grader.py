# In this file, we're going to implement a chain that will grade the answer and will determine whether this answer answers the questions or not.

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


# And you might be thinking, hey, why I'm not putting this into a note.
# A note that will check grounding in the documents, and a note that will check grounding in the question.
# Or maybe to have them both into a note and then decide what to do.
# And the answer is, of course I can do it if I want.
# However, because in this process, we do decide whether we want to finish or we want to generate it
# again, or maybe to perform another search, then this heavily implies that we need to choose the next step.
# So obviously here conditional branching sounded more intuitively to me.
# In another meaning, we don't put this chain in a node because we don't have much logic to create a node for.
# So we will use/put this chain directly in the graph workflow. 




class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


llm = ChatOllama(temperature=0, model="qwen3:8b")

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
