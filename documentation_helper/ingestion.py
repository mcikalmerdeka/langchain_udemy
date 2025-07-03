import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv("../.env")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from firecrawl import FirecrawlApp, ScrapeOptions

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

# Define a function to ingest the data using Firecrawl
def ingest_docs_firecrawl() -> None:
    langchain_documents_base_urls = [
        "https://python.langchain.com/docs/integrations/chat/",
        "https://python.langchain.com/docs/integrations/llms/",
        "https://python.langchain.com/docs/integrations/text_embedding/",
        "https://python.langchain.com/docs/integrations/document_loaders/",
        "https://python.langchain.com/docs/integrations/document_transformers/",
        "https://python.langchain.com/docs/integrations/vectorstores/",
        "https://python.langchain.com/docs/integrations/retrievers/",
        "https://python.langchain.com/docs/integrations/tools/",
        "https://python.langchain.com/docs/integrations/stores/",
        "https://python.langchain.com/docs/integrations/llm_caching/",
        "https://python.langchain.com/docs/integrations/graphs/",
        "https://python.langchain.com/docs/integrations/memory/",
        "https://python.langchain.com/docs/integrations/callbacks/",
        "https://python.langchain.com/docs/integrations/chat_loaders/",
        "https://python.langchain.com/docs/concepts/",
    ]

    # Just for demo purposes we will only ingest the first URL
    langchain_documents_base_urls2 = [langchain_documents_base_urls[0]]
    
    # Initialize Firecrawl app
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    for url in langchain_documents_base_urls2:
        print(f"FireCrawling {url=}")
        
        # Crawl using official Firecrawl SDK
        crawl_result = app.crawl_url(
            url,
            limit=10,
            scrape_options=ScrapeOptions(
                formats=['markdown', 'html'],
                only_main_content=True
            )
        )
        
        # Convert Firecrawl results to LangChain Documents
        docs = []
        if crawl_result and 'data' in crawl_result:
            for item in crawl_result['data']:
                doc = Document(
                    page_content=item.get('markdown', ''),
                    metadata=item.get('metadata', {})
                )
                docs.append(doc)

        print(f"Going to add {len(docs)} documents to Pinecone")
        
        if docs:  # Only add if we have documents
            PineconeVectorStore.from_documents(
                docs, embeddings, index_name="firecrawl-langchain-index"
            )
            print(f"****Loading {url} to vectorstore done ***")
        else:
            print(f"No documents found for {url}")

if __name__ == "__main__":
    ingest_docs_firecrawl()