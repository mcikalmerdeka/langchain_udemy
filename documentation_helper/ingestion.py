import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

# Create function to ingest the data
def ingest_docs():
    # Load the data using DirectoryLoader with UTF-8 encoding
    print("Ingesting data...")
    
    # Custom TextLoader with UTF-8 encoding
    class UTF8TextLoader(TextLoader):
        def __init__(self, file_path: str):
            super().__init__(file_path, encoding='utf-8')
    
    loader = DirectoryLoader(
        "documentation_helper/langchain-docs-0.2.6/",
        glob="**/*.html",
        loader_cls=UTF8TextLoader,
        show_progress=True
    )
    raw_documents = loader.load()
    print("Data ingested successfully")
    print(f"Loaded {len(raw_documents)} documents")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50, add_start_index=True)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} chunks")

    ## Update the metadata for each document
    for doc in documents:
        new_url = doc.metadata["source"].replace("documentation_helper/langchain-docs-0.2.6/", "https://")
        doc.metadata["source"] = new_url

    index_name = os.environ.get("PINECONE_INDEX_NAME")
    print(f"Inserting {len(documents)} documents into Pinecone index '{index_name}'...")

    # Initialize empty PineconeVectorStore
    vector_store = PineconeVectorStore(embedding=embeddings, index_name=index_name)
    
    # Process documents in batches to avoid Pinecone size limits
    batch_size = 100
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Uploading batches"):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
        
        # Add batch to vector store
        vector_store.add_documents(batch)

    print("Data ingested successfully")


if __name__ == "__main__":
    ingest_docs()