from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer
from typing import List

load_dotenv()

# Define the base tavily tool
def run_queries(search_queries: List[str], **kwargs):
    """Run the generated queries"""
    tavily_tool = TavilySearch(max_results=3)
    results = tavily_tool.batch([{"query": query} for query in search_queries]) # batch is a method that allows us to run concurrently
    return results

# The tool node going to run 2 different tools:
# 1. Search tool that originates from the first creation of the research
# 2. Search queries tool that originates from the revision of the research
execute_tools = ToolNode(
    tools=[
        StructuredTool.from_function(func=run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(func=run_queries, name=ReviseAnswer.__name__)
    ]
)