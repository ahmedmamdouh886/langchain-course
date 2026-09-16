from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tools


# Note: the revise node is gonna run infinitely because the LLM hallucinates, I don't know the problem yet.

# This means we want only two iterations of revision, draft, revision, draft and we want to limit this.
# This is not the best practice way, we can add another node(could be llm) as a judge instead of this iteration.
MAX_ITERATIONS = 2

# This is the responder node in the arch.png.
# So this node is going to receive the state,
# which I remind you only has a list of messages.
# And we simply want to draft the response.
# So this function is going to take all of the messages that we have in the graph and send it.
# And when we start to execute this graph, the first message is going to be the human message with our user input.
# So when we're going to get a response from the first responder,
# we're going to be getting an answer question(AnswerQuestion class).
# This is going to be the type of the response.
# So it's going to have an answer, it's going to have the reflection,
# so what needs to change, what is "missing",
# what is "superfluous", and the "search_queries".
# So these are going to be the search queries which we want to run later.
# And lastly, once we get this response, we want to append it into our messages key in our state.
# So this line over here return {"messages": [response]} is going to take the message and it's going to append it to the state.
def draft_node(state: MessagesState):
    """Draft the initial response."""

    response = first_responder.invoke({"messages": state["messages"]})
    
    return {"messages": [response]}


# This is the revisor node in the arch.png.
# This node is going to revise the answer based on the tool results, which was the Pydantic critique.
# And here, we simply want to take our reviser chain and we want to invoke it with all the messages we have so far.
# And in the first iteration of this node, it's going to be the result of the the draft_node node.
def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""

    response = revisor.invoke({"messages": state["messages"]})
    
    return {"messages": [response]}



def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count."""
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    
    num_iterations = count_tool_visits
    
    if num_iterations > MAX_ITERATIONS:
        return END

    return "execute_tools"




builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])

graph = builder.compile()

print(graph.get_graph().draw_mermaid())


res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
            }
        ]
    }
)

# Extract the final answer from the last message with tool calls.
last_message = res["messages"][-1]

if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])

print(res)

