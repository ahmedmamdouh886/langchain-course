import datetime

from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)


from langchain_core.messages import HumanMessage
# ChatPromptTemplate will hold all of our history of agent iterations. 
# MessagesPlaceholder for message placeholder.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from schemas import AnswerQuestion, ReviseAnswer


llm = ChatOllama(temperature=0, model="gpt-oss:20b")


# Note that we use function calling and output parser to get structured output to format the llm response and AnswerQuestion and ReviseAnswer is the format layer.



# We're going to create two output parsers.
# JsonOutputToolsParser is simply going to return us the function call we got back from the LLM and transform it into a dictionary.
# PydanticToolsParser is going to take the response from the LLM, it's going to search for the function-calling invocation, and it's going to parse it and transform it into an AnswerQuestion object.
# So, it's going to take the answer from the LLM and it's going to create an AnswerQuestion object that we can easily work with.
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

# This is the system prompt.
# This will hold the main prompt and all of our history we used to get so far.
# The prompt number 2 will be used by the Revisor node.
# The prompt number 3 is the is the search query and it will be used by the tool execution node which is Tavily search engine.
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement. 
3. Recommend search queries to research information and improve your answer.
4. You must return the information of your critique of what is missing in a field called "missing" and the information that don't add up in a field called "superfluous".
""",
        ),
        MessagesPlaceholder(variable_name="messages"), # This will hold all of our history so far. So all the information of what to search and what was critiqued is gonna be here.
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)


revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer") # tool_choice is currently ignored as it is not supported by Ollama.


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
        " list startups that do that and raised capital."
    )

    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") # tool_choice is currently ignored as it is not supported by Ollama.
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]}) # messages here is the MessagesPlaceholder(variable_name="messages") we defined above.

    print(res)
