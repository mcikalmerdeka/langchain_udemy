from backend.core import run_llm
import streamlit as st
from dotenv import load_dotenv
from typing import Set

load_dotenv("../.env")

# Set the header title of the app
st.header("LangChain Documentation Helper")

#  Set the prompt input for the user to ask a question
prompt = st.text_input("Prompt", placeholder="Enter your question here:")

# Define several session states for memory in chat history
if ("user_question_history" not in st.session_state
    and "chat_answer_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["user_question_history"] = []
    st.session_state["chat_answer_history"] = []
    st.session_state["chat_history"] = []


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

# If the user has entered a prompt, generate a response
if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])

        source_documents = [doc.metadata["source"] for doc in generated_response["source_documents"]]

        # Format the response with the sources
        formatted_response = (
            f"{generated_response['result']}\n\n {create_sources_string(source_documents)}"
        )

        # Append the user question and chat answer to the session state
        st.session_state["user_question_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

# Create a chat interface to display the chat history
if st.session_state["user_question_history"] and st.session_state["chat_answer_history"]:
    for gen_question, gen_answer in zip(st.session_state["user_question_history"], st.session_state["chat_answer_history"]):
        # Display the user question and chat answer in the chat interface
        st.chat_message("user").write(gen_question)
        st.chat_message("assistant").write(gen_answer)
