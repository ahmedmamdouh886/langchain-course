from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch

# Used to convert a python function to a tool that can be used by LLM.
# So it's going to take that function and provide to the LLM a structured schema for the function, which
# will help the LLM understand how to use this tool.
# And it's going to look in the state for the messages key.
# It's going to check the last message, and it's then going to see if there's any tool calls that were decided by the alarm.
# And if there are, it's going to execute those tools for us.
# And it can do this even in parallel. So this saves us tons of work.
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer


tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""

    # batch means run concurrently.
    return tavily_tool.batch([{"query": query} for query in search_queries])


# So we want to take the original tool and its functionality, and we want to create from it two different tools.
# It's going to be two different tools with the same functionality of the search, but they're going to
# have different names because they serve different purposes in the application workflow.
# So we're going to have an answer question tool(AnswerQuestion), which is going to be used during the initial research
# phase when the agent is first answering the question.
# And then we want to have this revised answer tool(ReviseAnswer) which is used during the revision phase when the agent
# is improving its answer based on the reflection.
# So both tools are going to run the search.
# Because we want to get their names in order to label those tools.
# But theoretically we can use one tool, but having separate names in two separate tools allows the system
# to clearly track which stage of the research process triggered the search initial research versus the revision research.
# So it's going to help us in debugging and evaluating the response.


# I remind you this tool(ToolNode) is going to examine the state.
# It's going to check the last message.
# And if there is a tool call it's going to execute the relevant tool call.

execute_tools = ToolNode(
    [
        # from_function is going to receive a function and convert it into
        # a tool with schema and description and all of that that we saw before.
        # And another argument is going to be the name of the tool.
        # So here the name is going to be the name of the class.
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
