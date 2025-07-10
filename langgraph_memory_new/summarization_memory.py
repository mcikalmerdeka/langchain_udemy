"""
This file is a example of how to use summarization logic in langchain to compress the previous conversation history.
The summarization logic will only kicks in when the total conversation history exceeds a certain threshold messages.
The messages that are deleted are the previous messages that have been summarized and the counter will reset to 1.
"""

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# Create graph
builder = StateGraph(state_schema=MessagesState)    

# Define the function for the chat node
def chat_node(state: MessagesState):
    system_message = SystemMessage(content="You're a kind therapy assistant.")
    
    # Get the history messages except the last human message
    history = state["messages"][:-1]

    # If the history exceeds the threshold, we need to summarize the previous messages
    if len(history) >= 8:
        print("🔄 Summarization logic triggered - compressing conversation history...")
        # Get the last human message
        last_human_message = state["messages"][-1]

        # Create the summary prompt
        summary_prompt = (
            "Distill the above chat messages into a single summary message. "
            "Include as many specific details as you can."
        )
        # Generate the summary of previous messages
        summary_message = llm.invoke(history + [HumanMessage(content=summary_prompt)])
        
        # Delete the previous messages because it have been summarized
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]

        # Create the human message
        human_message = HumanMessage(content=last_human_message.content)

        # Generate the response using the summary message and the last human message
        response = llm.invoke([system_message, summary_message, human_message])

        # Update the messages
        message_updates = [summary_message, human_message, response] + delete_messages
    else:
        # If the history does not exceed the threshold, we can just generate the response using the last human message
        message_updates = llm.invoke([system_message] + state["messages"])

    return {"messages": message_updates}

# Define the nodes and edges
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")

# Compile graph with MemorySaver and run the chat
memory = MemorySaver()
chat_app = builder.compile(checkpointer=memory)

# Run the chat
if __name__ == "__main__":
    thread_id = 1

    # Create a loop for the chat
    while True:
        # Takes the user input, wrap it in a HumanMessage, and append list of messages stored
        user_input = input("You: ")
        state_update = {"messages": [HumanMessage(content=user_input)]}
        
        result = chat_app.invoke(
            state_update,
            {"configurable": {"thread_id": thread_id}}
        )

        ai_msg = result["messages"][-1]
        print("Bot:", ai_msg.content)

    
