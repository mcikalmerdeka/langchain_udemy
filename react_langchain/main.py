import os
from dotenv import load_dotenv
load_dotenv()

from typing import Union, List

from langchain.agents import tool
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents.format_scratchpad import format_log_to_str
from callbacks import AgentCallbackHandler
    
# ReActSingleInputOutputParser: Parses LLM output that follows the ReAct (Reasoning and Acting) pattern
# It converts raw text responses into structured AgentAction or AgentFinish objects
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser

# AgentAction: Represents when the agent wants to use a tool (contains tool name, input, and reasoning)
# AgentFinish: Represents when the agent has a final answer (contains the final response)
from langchain_core.agents import AgentAction, AgentFinish

# Define tools
@tool
def get_text_length(text: str) -> int:
    """
    Returns the lenght of a text by character
    """
    print(f"get_text_length tool called with text: {text}")
    text = text.strip("'\n'").strip("'")
    return len(text)

# Function to find a tool by name
def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    """
    Find a tool by name
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

if __name__ == "__main__":
    print("Hello ReAct Langchain!")

    tools = [get_text_length]

    # This template follows the ReAct pattern: Reasoning (Thought) + Acting (Action)
    # The LLM must respond in this specific format for the parser to work correctly
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
    Thought: {agent_scratchpad}
    """

    # Create a prompt template
    prompt = PromptTemplate.from_template(template).partial(
        tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools])
    )

    # Initialize the callbacks
    callbacks = [AgentCallbackHandler()]

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, 
                     api_key=os.getenv("OPENAI_API_KEY"),
                     stop=["\nObservation:"], # Stop the LLM when it sees the word "Observation:"
                     callbacks=callbacks)

    intermediate_steps = []
    
    # Create an agent pipeline using LangChain's pipe operator
    # The pipeline: input -> prompt template -> LLM -> output parser
    # 
    # ReActSingleInputOutputParser is the key component that:
    # 1. Takes raw text output from the LLM
    # 2. Uses regex to parse two possible formats:
    #    - Action format: "Thought: ... Action: tool_name Action Input: input" -> returns AgentAction
    #    - Final format: "Thought: ... Final Answer: result" -> returns AgentFinish
    # 3. Converts the text into structured objects that the agent loop can use
    agent = (
        {
            "input": lambda x: x["input"],
             "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"])
        } 
        | prompt 
        | llm 
        | ReActSingleInputOutputParser()
    )

    agent_step = None

    # Create a loop to invoke the agent until it returns an AgentFinish
    iteration = 1
    while agent_step is None or not isinstance(agent_step, AgentFinish):
        print(f"\n=== ITERATION {iteration} ===")
        iteration += 1

        # Invoke the agent with a test question
        # Pass "intermediate_steps" since that's what the lambda function expects
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of the word 'Jokowi' in characters?",
                "intermediate_steps": intermediate_steps
            }
        )

        # Check if the agent decided to take an action (use a tool)
        # This works because ReActSingleInputOutputParser converted the raw LLM text into a structured AgentAction object when it detected the Action/Action Input pattern
        if isinstance(agent_step, AgentAction):
            
            # Extract the tool name the agent wants to use (parsed from "Action: tool_name")
            tool_name = agent_step.tool
            
            # Find the actual tool function from our tools list
            tool_to_use = find_tool_by_name(tools, tool_name)
            
            # Get the input the agent wants to pass to the tool (parsed from "Action Input: input")
            tool_input = agent_step.tool_input

            # Execute the tool with the agent's input and get the result
            observation = tool_to_use.invoke(tool_input)
            print(f"Observation: {observation=}")
            
            # Add the tool call and its result to the intermediate steps
            intermediate_steps.append((agent_step, str(observation)))

    # Final check if we returned an AgentFinish output
    if isinstance(agent_step, AgentFinish):
        print(f"Final Agent step (This will now output as AgentFinish): {agent_step.return_values['output']}")