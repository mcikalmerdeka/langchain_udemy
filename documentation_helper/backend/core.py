from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain import hub
import os

load_dotenv()

# Define the index name
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

# Define the retrieval chain
def run_llm(query: str):
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
    
    stuff_document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=stuff_document_chain) 
    return retrieval_chain.invoke({"input": query})

if __name__ == "__main__":
    result = run_llm("What is a LangChain chain?")
    print(result["answer"])