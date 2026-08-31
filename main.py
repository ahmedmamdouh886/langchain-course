from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


llm = ChatOllama(temperature=0, model="gpt-oss:20b")
tools = [TavilySearch()]
agent = create_agent(model=llm,tools=tools)

def main():
    print("Hello from langchain-course!")
    # result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")})
    result = agent.invoke({"messages":HumanMessage(content="What is the weather now in Tokyo?")})
    print(result)


if __name__ == "__main__":
    main()
