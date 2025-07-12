import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent

load_dotenv()

def main():
    print("Start...")

    # Define the instructions for the agent and the prompt
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    # Define the LLM for the agent
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    # Define the tools for the agent
    tools = [PythonREPLTool()]

    # # Create the agent for qr code generation
    # agent = create_react_agent(
    #     prompt=prompt,
    #     llm=llm,
    #     tools=tools,
    # )

    # # Create the agent executor for qr code generation
    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # # Invoke the agent executor for qr code generation
    # agent_executor.invoke(
    #     input={
    #         "input": r"""Generate and save in this directory E:\Udemy\LangChain- Develop LLM powered applications with LangChain\code_interpreter\qrcodes 15 QRcodes
    #                     that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    #     }
    # )

    # Create the agent for csv file
    csv_agent = create_csv_agent(
        llm=llm,
        path="code_interpreter/data/episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    # Invoke the agent for csv file
    csv_agent.invoke(
        # input={"input": "how many columns are there in file episode_info.csv"}
        # input={"input": "how many titles are there in the file episode_info.csv"}

        # For this one we will get Larry David with 49 episodes, but the answer should be 58 episodes
        # This happens because several episodes are written by multiple writers, so as episode with single writer, Larry David is indeed the correct answer
        input={"input": "in the file episode_info.csv, which writer wrote the most episodes? and how many episodes did they write?"}


        # input={"input": "print the seasons by ascending order of the number of episodes they have"}
    )


if __name__ == "__main__":
    main()


