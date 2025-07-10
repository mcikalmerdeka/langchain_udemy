"""
This file is a simple example of how to use the MemorySaver to store the conversation history in the prompt window.
Prompt window just means the context window of the LLM, the best one at the moment (July 2025) are gpt-4.1 and other models that have 1M tokens context window.
"""

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# Initialize the graph
builder = StateGraph(state_schema=MessagesState)

# Define the function for the chat node
def chat_node(state: MessagesState):
   system_message = SystemMessage(content="You're a kind therapy assistant.")

   # Get the history
   history = state["messages"]

   # Create the prompt with the system message and the all chat history
   prompt = [system_message] + history

   # Generate the response
   response = llm.invoke(prompt)
   return {"messages": response}

# Define the nodes
builder.add_node("chat", chat_node)

# Define the edges
builder.add_edge(START, "chat")

# Define the memory for storing the state in memory
memory = MemorySaver()

# Compile the graph
chat_app = builder.compile(checkpointer=memory)

# Run the chat
if __name__ == "__main__":
    thread_id = 1

    #  Create a loop for the chat
    while True:

        # Takes the user input, wrap it in a HumanMessage, and append list of messages stored
        user_input = input("You: ")
        state_update = {"messages": [HumanMessage(content=user_input)]}
        
        # Invoke the chat app with the state update and the thread id
        result = chat_app.invoke(
            state_update,
            {"configurable": {"thread_id": thread_id}}
        )
        
        # print(result) # For debugging

        # Get the last message from the result for the AI response
        ai_msg = result["messages"][-1]
        print("Bot:", ai_msg.content)

        # Update the thread id
        thread_id += 1

        # Quit the chat
        if user_input.lower() == "quit":
            break