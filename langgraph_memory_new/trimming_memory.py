"""
This file is a example of how to use trim_message method in langchain 
so that the conversation history from certain k number of messages is summarized and compressed.
"""

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import trim_messages
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# Create a trimmer that will compress the conversation history
# Note that max_tokens here means that the trimmer will keep the last 10 tokens of the conversation history (5 pairs of human + ai messages)
trimmer = trim_messages(strategy="last", max_tokens=10, token_counter=len)

# Create graph
builder = StateGraph(state_schema=MessagesState)    

# Define the function for the chat node
def chat_node(state: MessagesState):
    system_message = SystemMessage(content="You're a kind therapy assistant.")
    
    # Trim the messages
    trimmed_messages = trimmer.invoke(state["messages"])

    # Create the prompt with the system message and the trimmed messages
    prompt = [system_message] + trimmed_messages

    # Generate the response
    response = llm.invoke(prompt)

    return {"messages": response}

# Define the nodes and edges
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")

# Compile graph with MemorySaver
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
        
        # Invoke the chat app with the state update and the thread id
        result = chat_app.invoke(
            state_update,
            {"configurable": {"thread_id": thread_id}}
        )

        # Get the last message from the result for the AI response
        ai_msg = result["messages"][-1]
        print("Bot:", ai_msg.content)

        # Update the thread id
        thread_id += 1

        # Quit the chat
        if user_input.lower() == "quit":
            break

    
