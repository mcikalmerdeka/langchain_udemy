from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv("../.env")

# Define the system message
SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""
# Create a node that will run the agent reasoning
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    # Create messages list: system message + unpacked user messages
    # *state.messages unpacks the list so messages are added individually, not as nested list
    messages = [{"role": "system", 
                 "content": SYSTEM_MESSAGE}, 
                 *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Create a node that will run the tool node
tool_node = ToolNode(tools)







