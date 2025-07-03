import asyncio

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1")

# Define the main function
async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [
                    "mcp_server/server/math_server.py"
                ],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)

    # Run the weather server
    result = await agent.ainvoke({"messages": "What is the weather in San Francisco?"})

    # # Run the math server
    # result = await agent.ainvoke(
    #     {"messages": "What is 2 + 2?"}
    # )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())