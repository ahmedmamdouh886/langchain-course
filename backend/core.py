from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model # A convient way to communicate with model provider.
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma



load_dotenv()

# 1.Initialize embeddings and vectore store (same as ingestion.py)
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)


# 2.Initialize chat model
model = init_chat_model("qwen3:8b", model_provider="ollama")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents
    # We can use here vectorstore.similarity_search(query, k=4), however we decided to use as_retriever() because it gives us better traceability in the langsmith.
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4) # Invoke method will perform similarity search.
   
    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    # Return both serialized content and raw documents
    # Note: this tool return content and artifact as the response_format parameter define, so the serialized is the content and the retrieved_docs is the artifact.
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """

    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so." # This line is so important, because if the documentation assistant doesn't know how to answer our question, we don't want it hallucinate.
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build messages list
    messages = [{"role": "user", "content": query}]
    
    # Invoke the agent
    response = agent.invoke({"messages": messages})
    
    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract context documents from ToolMessage artifacts
    # To show the user references of the documents we answered his question from (Returned from the retrieve_context tool we defined above with @tool(response_format="content_and_artifact") ), to build a trust between us and the end user.
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }



if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)

