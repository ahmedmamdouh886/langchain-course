from dotenv import load_dotenv

from langchain_core.messages import HumanMessage

# MessagesState for preserving state, StateGraph for initializing the graph.
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoning, tool_node

load_dotenv()


AGENT_REASON="agent_reason"
ACT= "act"
# And this is for future when we're going to reference the last message between the user and the agent.
# And the last message is going to be at the -1 index. So this is some Python notation.
# And again this is simply for more readability when we write the code. So the code is a bit cleaner.
LAST = -1


# ---- Implementing our graph ---- #


# 1. Graph initialization.
flow = StateGraph(MessagesState)


# 2. Add nodes to the graph.
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

# This will return what node should run after the AGENT_REASON node, END or ACT.
def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    
    return ACT


# 3. Add an edge from AGENT_REASON to ACT and END nodes. This is a conditional nodes, so it's up to the AGENT_REASON node to decide to go to ACT or END node.
# This is the dotted lines in the flow image, one for END node and the other for ACT node.
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END: END,
    ACT: ACT
})

# 4. Add another edge from ACT to AGENT_REASON, Because after we invoke the tool we want the agent to reason and to figure out whether it needs to return the answer or to run another tool call.
# This is the line from ACT to AGENT_REASON in the flow image.
flow.add_edge(ACT, AGENT_REASON)

# 5. Run the graph flow.
app = flow.compile()

# 6. Draw the flow we built.
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


if __name__ == "__main__":
    print("Hello Langgraph")
