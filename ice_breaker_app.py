import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

# Create function to get the Linkedin data
def ice_break_with(name: str) -> str:
    linkedin_profile_url = linkedin_lookup_agent(name=name)

    # Get the Linkedin data from third-party tool
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    # Create a prompt template
    summary_template = """
    Given the Linkedin information {information} about a person, I want you to create two things:

    1. A short summary of the person's life
    2. Two interesting facts about the person
    """

    prompt = PromptTemplate(
        input_variables=["information"], # The input variables for the prompt
        template=summary_template,       # The template for the prompt
    )

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    # Create a chain (chain of prompts and LLMs with output parsers)
    chain = prompt | llm | StrOutputParser()

    # Run the chain
    res = chain.invoke(input={"information": linkedin_data})
    print(res)

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    print("Ice breaker app is running...")

    # Run the ice breaker app
    ice_break_with(name="Cikal Merdeka")