import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv("../.env")

# Define a custom tool
@tool
def triple(num: float) -> float:
    """
    Returns the triple of the input.

    Args:
        num (float): The input number.

    Returns:
        float: The triple of the input.
    """
    return num * 3

# Create a list of tools including a TavilySearch tool
tools = [TavilySearch(max_results=1), triple]

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4.1", 
                 temperature=0, 
                 api_key=os.getenv("OPENAI_API_KEY")
).bind_tools(tools) # Using this method, we dont need to handle any parsing because the LLM vendors will handle it for us