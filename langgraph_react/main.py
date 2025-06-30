import os
from dotenv import load_dotenv
from langgraph.graph import MessagesState, StateGraph, START, END
from nodes import run_agent_reasoning, tool_node

load_dotenv("../.env")

# Define the nodes names
AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1 # Last index of the messages list

# Define a function to check if the last message is a tool use or a final answer, this will be used for the agent reasoning node
def should_continue(state: MessagesState) -> str:
    """
    Check if the last message is a tool use or a final answer.
    """
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

# Create a graph with edges
graph = StateGraph(MessagesState)
graph.add_node(AGENT_REASON, run_agent_reasoning)
graph.add_node(ACT, tool_node)

# Set up the entry point using START
graph.add_edge(START, AGENT_REASON)

# Set up the conditional edges
graph.add_conditional_edges(AGENT_REASON, should_continue, {
    END: END,
    ACT: ACT
})

# Set up the edge from the act node to the agent reasoning node
graph.add_edge(ACT, AGENT_REASON)

# Compile the graph and draw it
app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="langgraph_react/react_graph.png")

if __name__ == "__main__":
    print(f"Hello ReAct LangGraph! with Function Calling!")