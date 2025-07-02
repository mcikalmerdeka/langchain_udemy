from backend.core import run_llm
import streamlit as st
from dotenv import load_dotenv
from typing import Set

load_dotenv("../.env")

# Set the header title of the app
st.header("LangChain Documentation Helper")

#  Set the prompt input for the user to ask a question
prompt = st.text_input("Prompt", placeholder="Enter your question here:")

# Create a function to format the sources string
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"- {source}\n"
    return sources_string

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt)

        source_documents = [doc.metadata["source"] for doc in generated_response["source_documents"]]

        formatted_response = (
            f"{generated_response['result']}\n\n {create_sources_string(source_documents)}"
        )

        st.markdown(formatted_response)