from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from typing import List, Dict, Any
import os

load_dotenv("../.env")

# Define the index name
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

# Define the retrieval chain
def run_llm(query: str, chat_history: List[Dict[str, Any]]):
    """Run the LLM with the given query."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.environ.get("OPENAI_API_KEY"))
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings
    )
    retriever = vector_store.as_retriever()
    
    llm = ChatOpenAI(model="gpt-4.1", verbose=True, temperature=0)
    
    # Define the prompt from the hub
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Define a new prompt to rephrase the user question based on the chat history
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    # Create a new retriever that is aware of the chat history
    history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=rephrase_prompt)

    stuff_document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_document_chain) 
    
    initial_result = retrieval_chain.invoke({"input": query, "chat_history": chat_history})

    new_result = {
        "query": initial_result["input"],
        "result": initial_result["answer"],
        "source_documents": initial_result["context"],
    }

    return new_result

if __name__ == "__main__":
    result = run_llm("What is a LangChain chain?")
    print(result["answer"])