import os
import datetime
from dotenv import load_dotenv

from langchain_core.output_parsers.openai_tools import ( # Leverage opeai function calling to structure the output
    JsonOutputToolsParser,
    PydanticToolsParser
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from schemas import Reflection, AnswerQuestion, ReviseAnswer

load_dotenv()

# Define the model
llm = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))

# Define the output parser
json_output_parser = JsonOutputToolsParser(return_id=True)
pydantic_output_parser = PydanticToolsParser(tools=[AnswerQuestion])

# Actor Prompt Template
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat()) # Populate the time variable with the current time

# First responser prompt template
first_responder_prompt_template = actor_prompt_template.partial(first_instruction="Provide a detailed ~250 word answer")

# First responder chain
first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    [AnswerQuestion], tool_choice="AnswerQuestion") # The tool_choice will force the model to use the AnswerQuestion tool

# Add revision instruction to the prompt template (this will be plugged into the {first_instruction} variable)
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

# Add revision chain
revise_chain = actor_prompt_template.partial(first_instruction=revise_instructions) | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == "__main__":
    print("Hello Langgraph Reflexion Agent")

    human_message = HumanMessage(
        content="Write about AI-powered SOC / autonomous SOC promblem domain, and list startups that do that and raised capital")
    
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | pydantic_output_parser
    )

    result = chain.invoke(input={"messages": [human_message]})
    print(result)