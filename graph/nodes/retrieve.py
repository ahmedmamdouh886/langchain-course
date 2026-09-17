# This is the retrieve node in the arch.png

from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

# This node will return a dictionary, it means it will return what to update in the GraphState.
def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")

    question = state["question"]

    documents = retriever.invoke(question) # This will do semantic search.

    # What to update in the GraphState.
    return {"documents": documents, "question": question}

