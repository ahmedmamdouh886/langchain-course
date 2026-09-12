from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama

load_dotenv()


# This file will hold our reasoning engine.

@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """

    return float(num) * 3


tools = [TavilySearch(max_results = 1), triple]

# There are many ways to define function calling to LLM, one via ReAct prompt and another one via LLM because LLMs has evolved nowadays, and bind_tools function let us define function calling to LLM, and this is ofcourse based on the LLM vendor.
llm = ChatOllama(temperature=0, model="qwen3:8b").bind_tools(tools) # To review bind_tools function review the lesson 93. [Hands On] Coding the Agent's Brain: Implementing the ReAct Runnable.

