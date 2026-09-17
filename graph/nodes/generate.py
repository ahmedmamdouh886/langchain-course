# This is the generate node in the arch.png, this node will recieve our documents after it filtered out by previous nodes, so we can send the them to the LLM to generate a response.

from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})

    # What to update in the GraphState.
    return {"documents": documents, "question": question, "generation": generation}

