import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()




if __name__ == "__main__":
    print("Hello From Langgraph Agentic RAG")
