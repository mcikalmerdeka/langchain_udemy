from dotenv import load_dotenv
from langchain_community.document_loaders.youtube import YoutubeLoader

load_dotenv()

if __name__ == "__main__":
    # YouTube URL from the user
    youtube_url = "https://youtu.be/_wHjDNzjF-k?si=lbinTrLyCfIx1NLB"
    
    print("Loading YouTube transcript...")
    
    # Create loader from YouTube URL
    loader = YoutubeLoader.from_youtube_url(youtube_url)
    
    # Load the transcript
    documents = loader.load()
    
    print(f"Found {len(documents)} document(s)")
    
    # Print the transcript
    for i, doc in enumerate(documents):
        print(f"\n--- Document {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}") 