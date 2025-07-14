from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# Get the prompt template from the hub
prompt_template = hub.pull("rlm/rag-prompt")

# Define the generation chain
generation_chain = prompt_template | llm | StrOutputParser()