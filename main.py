# TypedDict is a dictionalry which will be used to generate structured dictionary, it will be used for state schema.
# Annotated is used to add metadata with type hints.
from typing import TypedDict, Annotated

from dotenv import load_dotenv

load_dotenv()


from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
# add_messages, the entire goal of this function is to ensure new messages are appended to the existing conversation history instead of replacing it.
# So all this function is doing is simply appending to a list here, and this is how it's going to be updating the state.
# add_messages is a "reducer"(reducer in LangGraph is a function that tells the framework how to combine an existing state value with a new update from a node).
# So the reducer by the way is a general terminology in graph. How do we want to update the state here.
# And this is one of Landgraf's key advantages when it comes to flexibility.
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain

# This the data structure that every node in the graph is gonna have access to, to update and manage state.
# And we want to keep updating these messages key after every iteration after every node execution in our graph.
# So we want to keep appending to this list because every execution is going to generate a message from the AI.
# And we want to go and append and append and append it right. So this is the goal of this state here.
# Simply a data structure to hold all of those list of messages here.
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})

    # Now notice that when we're going now to update the state when we're going to append to the messages key here we are appending now a human message.
    # So the AI response we're going to get is going to be an AI message.
    # But we are now casting this message into a human message. So this is a heuristic we're making.
    # So we want the critique that the LLM generates when we plug it back to the LLM, we want the LLM to think that this critique was the output of a user. So a human wrote it.
    # And the idea behind this is that large language models are also trained for conversation and for human feedback.
    return {"messages": [HumanMessage(content=res.content)]}


# Now it's important to note that in line graph, every node should receive as an input the state which should be of the type of the state that we initialize our graph with.

# We have one deterministic edge from the reflect node to the generate node.
# And then we have a conditional edge from the generate node either to the reflect node or the end node.
# And you can see it in the diagram as a dashed arrow here.

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


# This function is going to be called every time after we run the "generation node".
# And the output is going to tell the graph where to go next, either to go to the "reflection node" or to go to "end everything".
# This function will be used for "conditional edge".
def should_continue(state: MessageGraph):
    # You can imagine that instead of this logic we can actually put here a large language model to decide where to go.
    # So it's going to make the decision whether we need to do some more iterations or we are satisfied with the result.
    # And that's the beauty of using land graph, because we as developers, we can define the flow, we can define which nodes are going to execute.
    # And we can put here an LLM to decide where to go in this flow.
    # In this example I gave the number six,
    # But I could have given any number here or written any other logic to determine whether, if we want to finish or to go and continue reflect here.
    if len(state["messages"]) > 6:
        return END

    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    print("Hello LangGraph")
    
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post
                                  """
            )
        ]
    }

    response = graph.invoke(inputs)
    
    print(response)
