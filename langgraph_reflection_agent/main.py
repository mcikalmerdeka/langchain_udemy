import os
from typing import List, Sequence
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import START, END, MessageGraph # MessageGraph is a StateGraph where every node receives a list of messages
from chains import generate_chain, reflect_chain

load_dotenv()

# Define the nodes names
GENERATE = "generate"
REFLECT = "reflect"

# Define a function to detrermine the next node to go to
def should_continue(state: List[BaseMessage]):
    """
    Check if the state has more than 6 messages already.
    If it does, we go to the END node.
    If it doesn't, we go to the REFLECT node.
    This is to limit the number of messages in the state to 6.
    """
    if len(state) > 6:
        return END
    return REFLECT

# Define what gonna happen in the generation node
def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})

# Define what gonna happen in the reflection node
def reflection_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": state})
    return [HumanMessage(content=res.content)] # In here we use the HumanMessage cause we want to trick the agent into thinking that the critique is from the user

# Define the graph
graph = MessageGraph() 
graph.add_node(GENERATE, generation_node) # Add the generation node to the graph
graph.add_node(REFLECT, reflection_node) # Add the reflection node to the graph
graph.set_entry_point(GENERATE) # Set the entry point to the generation node

# Set up the conditional edges
graph.add_conditional_edges(GENERATE, should_continue, {
    END: END,
    REFLECT: REFLECT
})
graph.add_edge(REFLECT, GENERATE) # Set up the edge from the reflection node to the generation node

# Compile the graph
app = graph.compile()
# print(app.get_graph().draw_mermaid()) # This will print the graph in mermaid format
app.get_graph().draw_mermaid_png(output_file_path="langgraph_reflection_agent/reflection_graph.png")

if __name__ == "__main__":
    print("Hello Langgraph Reflection Agent")

    inputs = [HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)]
    response = app.invoke(inputs)

