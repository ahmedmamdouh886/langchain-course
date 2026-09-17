# Here in this file, we're going to be implementing a version of Adaptive Rack.
# And to be honest, it's a fancy word of simply using a question router to route our question to different rack flows.
# We're going to be using two rack flows.
# So the first one is going to be taking the route to search on the internet and then to continue downstream on the same logic we have before.
# And the second round is going to be to use the retrieval augmentation from our vector store.
# So we're first going to take the user's question, and we're going to decide whether the information
# is stored in the vector store to answer that question.
# And if it's not we're simply going to take the route to web search and answer from there.
# And the main thing we're going to do is to implement a question router chain, which will take the question
# and decide whether we're going to route it to the web search or to the retrieve node.


# And if you're not familiar with the literal type like I was a couple of months ago, then it provides
# a way to specify that a variable can only take one of predefined set of values.
# So this is very useful for validation and type checking.
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Used to structure LLM response.
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


llm = ChatOllama(temperature=0, model="qwen3:8b")

structured_llm_router = llm.with_structured_output(RouteQuery)


system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
