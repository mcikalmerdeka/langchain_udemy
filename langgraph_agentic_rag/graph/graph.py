import os
from dotenv import load_dotenv
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEB_SEARCH, GENERATE
from graph.nodes import retrieve_node, grade_documents_node, web_search_node, generate_node
from graph.state import GraphState
from langgraph.graph import START, StateGraph, END

load_dotenv()

# Define the function to handle conditional for grade documents node to web search node or generate node
def decide_to_generate(state: GraphState) -> bool:
    """
    Check if the documents are relevant to the question
    If not, set the web_search flag to True
    """
    print("---ASSES GRADED DOCUMENTS---")

    # Define the condition to check if the documents are relevant to the question
    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO THE QUESTION---")
        print("---ROUTING TO WEB SEARCH NODE---")
        return WEB_SEARCH
    else:
        print("---DECISION: ALL DOCUMENTS ARE RELEVANT TO THE QUESTION---")
        print("---ROUTING TO GENERATE NODE---")
        return GENERATE

# Define the graph
graph = StateGraph(GraphState)

# Add the nodes to the graph
graph.add_node(RETRIEVE, retrieve_node)
graph.add_node(GRADE_DOCUMENTS, grade_documents_node)
graph.add_node(GENERATE, generate_node)
graph.add_node(WEB_SEARCH, web_search_node)

# Add the edges to the graph
graph.add_edge(START, RETRIEVE)
graph.add_edge(RETRIEVE, GRADE_DOCUMENTS)
graph.add_conditional_edges(
    source=GRADE_DOCUMENTS,
    path=decide_to_generate,
    path_map={
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE
    }
)
graph.add_edge(WEB_SEARCH, GENERATE)
graph.add_edge(GENERATE, END)

# Compile the graph and save the graph to png
rag_app = graph.compile()
rag_app.get_graph().draw_mermaid_png(output_file_path="langgraph_agentic_rag/complete_rag_graph.png")