import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

# Create function to ingest the data
def ingest_docs():
    # Load the data
    print("Ingesting data...")
    loader = DirectoryLoader(
        "documentation_helper/langchain-docs-0.2.6/",
        loader_cls=UnstructuredHTMLLoader,
        show_progress=True,
        use_multithreading=True,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_documents = loader.load()
    print("Data ingested successfully")
    print(f"Loaded {len(raw_documents)} documents")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} chunks")

    index_name = os.environ.get("PINECONE_INDEX_NAME")
    print(f"Inserting {len(documents)} documents into Pinecone index '{index_name}'...")

    # Initialize PineconeVectorStore to add documents to it
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # Define the batch size
    batch_size = 100

    # Iterate over documents in batches and add them to Pinecone (useful approach for large documents)
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i : i + batch_size]
        vector_store.add_documents(batch)

    print("Data ingested successfully")


if __name__ == "__main__":
    ingest_docs()


