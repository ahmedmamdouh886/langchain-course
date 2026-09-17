# Now in this file we're going to implement a chain that is going to determine whether the answer we get back from the LLM, the generation node is grounded in the documents.


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


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


llm = ChatOllama(temperature=0, model="qwen3:8b")

# So basically the answer we will get back from the LM langChain will format it as the "pydantic class"
# of GradeHallucinations which we created above, which is going to have only one attribute of binary_score.
structured_llm_grader = llm.with_structured_output(GradeHallucinations)


system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
