import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv("../.env")

if __name__ == "__main__":
    print("Starting the application...")

    # Initialize the embeddings and LLM
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))

    # Testing the query
    query = "What is Pinecone in machine learning?"

    # # Create the chain (without Retrieval Vector Database)
    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    # Initialize the vector store
    vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

    # Create the new retrieval chain (with Retrieval Vector Database)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt) # This method will take a list of documents and combine them into a single prompt then pass it to the LLM
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)

    # Run the retrieval chain
    result = retrieval_chain.invoke({"input": query})
    print(result["answer"])

    