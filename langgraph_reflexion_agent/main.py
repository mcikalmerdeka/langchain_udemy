from dotenv import load_dotenv
from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import MessageGraph, START, END

from chains import first_responder_chain, revise_chain
from tool_executor import execute_tools

load_dotenv()

# Define parameter and node names   
MAX_ITERATIONS = 2
FIRST_RESPONDER = "first_responder"
EXECUTE_TOOLS = "execute_tools"
REVISE = "revise"

# Define function after the revision: whether to execute the tools again then revise or finish
def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    if count_tool_visits >= MAX_ITERATIONS:
        return END
    return EXECUTE_TOOLS

# Define the graph
graph = MessageGraph()
graph.add_node(FIRST_RESPONDER, first_responder_chain)
graph.add_node(EXECUTE_TOOLS, execute_tools)
graph.add_node(REVISE, revise_chain)

# Set up the edges
graph.add_edge(FIRST_RESPONDER, EXECUTE_TOOLS)
graph.add_edge(EXECUTE_TOOLS, REVISE)
graph.add_conditional_edges(REVISE, event_loop)

# Set up entry point
graph.add_edge(START, FIRST_RESPONDER)

# Compile the graph
app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="langgraph_reflexion_agent/reflexion_graph.png")

if __name__ == "__main__":
    print("Hello Langgraph Reflexion Agent")

    result = app.invoke("Write about AI-powered SOC / autonomous SOC promblem domain, and list startups that do that and raised capital")
    print(result)