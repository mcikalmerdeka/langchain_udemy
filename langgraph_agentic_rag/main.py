import os
from dotenv import load_dotenv
from graph.graph import rag_app

load_dotenv()

if __name__ == "__main__":
    print("Hello From Langgraph Agentic RAG Application")

    # # Experiment 1: Agent memory (inside the knowledge store)
    # print(rag_app.invoke(input={"question": "What is agent memory?"}))

    # # Experiment 2: Definition of Microsoft AI search service (outside the knowledge store, need to run web search node)
    # print(rag_app.invoke(input={"question": "what is the definition of Microsoft AI search service?"}))

    # Experiment 3: Places to visit in Indonesia (outside the knowledge store and out of topic, need to run web search node)
    print(rag_app.invoke(input={"question": "what are the places to visit in Indonesia?"}))