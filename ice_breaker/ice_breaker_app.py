import os
from typing import Tuple
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser # Only used if we dont use pydantic for output parsing
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_Agent import lookup as twitter_lookup_agent
from output_parsers import summary_parser, Summary

# Create function to get the Linkedin data
def ice_break_with(name: str) -> Tuple[Summary, str]:

    # Get the Linkedin information
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)

    # Get the Twitter information
    twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets(username=twitter_username)

    # Create a prompt template
    summary_template = """
    Given the information about a person from Linkedin {linkedin_data} and Twitter posts {tweets}, 
    I want you to create a short summary of the person's life.

    1. A short summary of the person's life
    2. Two interesting facts about the person

    \n\n
    Make sure to follow the format instructions strictly!
    {format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_data", "tweets"],                                        # The input variables for the prompt
        template=summary_template,                                                          # The template for the prompt
        partial_variables={"format_instructions": summary_parser.get_format_instructions()} # The format instructions for the prompt
    )

        # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    # Create a chain (chain of prompts and LLMs with output parsers)
    chain = summary_prompt_template | llm | summary_parser

    # Run the chain
    res: Summary = chain.invoke(input={"linkedin_data": linkedin_data, 
                              "tweets": tweets})
    
    return res, linkedin_data.get("photoUrl")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    print("Ice breaker app is running...")

    # Run the ice breaker app
    name = input("Enter the name of the person you want to ice break with: ")
    
    if name == "default":
        name = "Eden Marco Udemy"
    
    ice_break_with(name=name)

    print("Ice breaker app is done!")
    
    # If using my profile (only works with Linkedin though)
    # ice_break_with(name="Cikal Merdeka")