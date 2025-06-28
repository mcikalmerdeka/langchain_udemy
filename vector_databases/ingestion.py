import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone

load_dotenv()

if __name__ == "__main__":
    # Load the data
    print("Ingesting data...")
    loader = TextLoader("vector_databases/mediumblog1.txt", encoding="utf-8")
    document = loader.load()
    print("Data ingested successfully")

    # Split the document into chunks
    print("Splitting data into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(document)
    print(f"Created {len(chunks)} chunks")
    print("Data split successfully")

    # Embed the chunks
    print("Embedding data...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    print("Data embedded successfully")

    # Store the chunks in the vector database
    print("Storing data in the vector database...")
    
    # Option 1: Using from_documents (current approach)
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings
    )
    
    # Option 2: Following documentation pattern exactly (alternative)
    # pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # index = pc.Index(os.getenv("INDEX_NAME"))
    # vectorstore = PineconeVectorStore(
    #     index=index,
    #     embedding=embeddings,
    #     text_key="text",
    #     namespace="mediumblog1",
    # )
    # vectorstore.add_documents(chunks)
    
    print("Data stored successfully in vector database!")