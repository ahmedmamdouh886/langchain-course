# Here we will hold graph state.
from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    # Of course, we want to have the question in our state, because we always want to reference it,
    # whether to determine if the documents retrieved are relevant
    # to the question or even to what's to search online.
    question: str
    # And of course we want to have the generation field, which is going to be the generated answer.
    generation: str
    # We want to have a Boolean flag that will tell us whether we need to search online for extra results or not.
    web_search: bool
    # We want to save the documents that are going to help us answer this question.
    # So those are going to be the retrieved documents or the documents that we get back from the search result.
    documents: List[str]
