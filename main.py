from dotenv import load_dotenv

load_dotenv()

from typing import List

# Pydantic is a data validation library in Python.
# It provides us functionalities like data parsing, serialization, automatic type validation, and more.
from pydantic import BaseModel, Field

from langchain.agents import create_agent # This function will run the ReAct paradigm behind the scene, check langsmith tracing to inspect it.
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


# <This logic is a layer for the agent response format instead of json.>
# The point of this class is to represent the source of the answer.
class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

# </This logic is a layer for the agent response format instead of json.>

# llm = ChatOllama(temperature=0, model="gpt-oss:20b")
llm = ChatOllama(temperature=0, model="qwen3:8b")

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    # result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")})
    result = agent.invoke({"messages":HumanMessage(content="What is the weather now in Tokyo?")})
    print(result) # Inspect the structure_response key. The current result will return None and it must return AgentResponse class, I don't know why!


if __name__ == "__main__":
    main()
