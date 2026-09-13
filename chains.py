# This file holds all of our prompts and chains that we're gonna be using in our graph.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Then this prompt is supposed to act as our critique.
# So it's going to review the output.
# And in this case is a Twitter post and it's going to criticize it.
# So it's going to say how it can be better and a suggestion to improve it.
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
# Now we want to put here a placeholder for other messages.
# And those are going to be the history messages that our agent is going to invoke and to criticize and
# get recommendations over and over again.
# So here you can see that we put a messages placeholder and the variable name is going to be messages.
# So when we initialize the reflection point we're going to plug in to this prompt in the messages.
# And it's going to contain a lot of messages of our history.
        MessagesPlaceholder(variable_name="messages"), 
    ]
)


# And in our agent architecture, the generation prompt is going to generate the tweets that are going
# to be revised over and over again after the feedback we get from the reflection prompt.
# So it's going to revise the tweet until it gets the perfect tweet.
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatOllama(temperature=0, model="qwen3:8b")

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
