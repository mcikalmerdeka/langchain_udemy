import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub
from tools.tools import get_profile_url_tavily

load_dotenv()

# Create a LinkedIn lookup agent
def lookup(name: str) -> str:

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    # Create a prompt template
    template = """
    Given the full name {name_of_person}, I want you to get back the Linkedin Profile URL. Your answer should only contain the URL and nothing else.
    """

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["name_of_person"]
    )

    # Create a tool list
    linkedin_url_lookup_tool = Tool(
        name="Crawl Google for Linkedin profile page Using Tavily Search",
        func=get_profile_url_tavily,

        # Note that the description is important since that's how the agent will know whether to use the tool or not
        description="Useful when you need to find a person's Linkedin profile URL"
    )

    tools = [linkedin_url_lookup_tool]

    # Create a react prompt
    react_prompt = hub.pull("hwchase17/react")

    # Create a react agent with llm
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )

    # Create an agent executor from the agent
    # Note: The agent executor is going to be responsible for orchestrating all of the components
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # Run the agent executor
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linkedin_profile_url = result["output"]
    return linkedin_profile_url

if __name__ == "__main__":
    # Only use this if you are running the script directly
    # This is to ensure that the script can find the tools.py file
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    linkedin_profile_url = lookup("Cikal Merdeka Bandung Jawa Barat")
    print(linkedin_profile_url)