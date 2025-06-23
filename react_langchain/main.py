import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import tool
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser

# Define a tool
@tool
def get_text_length(text: str) -> int:
    """
    Returns the lenght of a text by character
    """
    print(f"get_text_length tool called with text: {text}")
    text = text.strip("'\n'").strip("'")
    return len(text)


if __name__ == "__main__":
    print("Hello ReAct Langchain!")

    tools = [get_text_length]

    # This template can be pulled from hwchase17/react, but we modified it to remove the agent_scratchpad
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:
    """

    # Create a prompt template
    prompt = PromptTemplate.from_template(template).partial(
        tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools])
    )

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, 
                     api_key=os.getenv("OPENAI_API_KEY"),
                     stop=["\nObservation:"]) # Stop the LLM when it sees the Observation: string
    
    # Create a chain
    agent = {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()
    res = agent.invoke({"input": "What is the length of the word 'Dog' in characters?"})
    print(res)
