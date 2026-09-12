# This file will hold the implementation of our langgraph nodes.

from dotenv import load_dotenv

# MessagesState is a simple object which has the dictionary of the key messages.
# And that messages is going to be a list of messages.
# So this is going to keep track on the state on all of the messages back in between our agent, the human message and the AI message.
from langgraph.graph import MessagesState

# This is a node.
# It's going to check the last message between the agent and human.
# And if that last message is an AI message that has a valid tool call, it's going to go and execute that
# tool code, assuming that tools is initiated with the tool node object.
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSYEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

# This is the agent_reason node depicted in the infrastructure image that will receive our message and decide wether to answer or Act(call a tool).
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    # Note, all the information will be in the *state["messages"] over here, the first iteration will have the human message, and the second iteration will have the human and the AI messages and so on.
    # So the state will be preserved here in the *state["messages"]
    response = llm.invoke([{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]])

    return {"messages": [response]} # So after the llm is invoked we need to update our state, that's why we appended the response to the messages key and returned it.

# We define the tool node with the relevant tools, this is depicted as Act node in the infrastructure image.
tool_node = ToolNode(tools)
