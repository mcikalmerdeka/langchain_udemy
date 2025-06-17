import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Define a function to get the content of a Wikipedia page
def get_wikipedia_content(url):
    """Fetch and extract text content from a Wikipedia page"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'table']):
            element.decompose()
        
        # Get the main content paragraphs
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if content_div:
            paragraphs = content_div.find_all('p')
            text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            return text[:5000]  # Limit to first 5000 characters to avoid token limits
        
        return "Could not extract content from the Wikipedia page"
    
    except requests.RequestException as e:
        return f"Error fetching the page: {str(e)}"

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# Create a prompt template
summary_template = """
Given the information {information} about a person, I want you to create two things:

1. A short summary of the person's life
2. Two interesting facts about the person
"""

prompt = PromptTemplate(
    input_variables=["information"], # The input variables for the prompt
    template=summary_template,       # The template for the prompt
)

# Create a chain (chain of prompts and LLMs with output parsers)
chain = prompt | llm | StrOutputParser()

# Fetch information from Wikipedia
wikipedia_url = "https://en.wikipedia.org/wiki/Elon_Musk"
information = get_wikipedia_content(wikipedia_url)

# Run the chain
res = chain.invoke(input={"information": information})
print(res)