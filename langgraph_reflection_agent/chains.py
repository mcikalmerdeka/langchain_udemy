import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Define the prompt template for the reflection agent
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (   # System message
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user to improve their tweet. "
            "Always detailed recommendations, including request for length, virality, style, etc."
        ),
        MessagesPlaceholder(variable_name="messages") # This is the placeholder for the history of messages for the agent to use
    ]
)

# Define the prompt template for the tweet generation agent
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts. "
            "Generate the best twitter post possible for the user's request. "
            "If the user provides critique, respond with a revised version of your previous attempts. "
            "Make sure your output contains only the revised tweet and nothing else, no need to response/greet to the person who provided the critique."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)   

# Define the LLM
llm = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))

# Define the chains for the critique and generation agents
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm



