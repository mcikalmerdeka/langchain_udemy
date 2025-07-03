from backend.core import run_llm
import streamlit as st
from dotenv import load_dotenv
from typing import Set
import time

load_dotenv()

# Configure page
st.set_page_config(
    page_title="LangChain Documentation Helper",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .user-info {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    
    .user-name {
        font-size: 1.2rem;
        font-weight: bold;
        color: #262730;
        margin-bottom: 0.5rem;
    }
    
    .user-email {
        font-size: 0.9rem;
        color: #666;
    }
    
    .profile-pic {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        margin-bottom: 1rem;
        border: 3px solid #ff6b6b;
    }
    
    .main-header {
        text-align: center;
        color: #262730;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 10px;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    
    .stTextInput > div > div > input {
        background-color: white;
        border-radius: 20px;
        border: 2px solid #e6e6e6;
        padding: 10px 15px;
    }
    
    .stButton > button {
        background-color: #ff6b6b;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #ff5252;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with user information
with st.sidebar:
    st.markdown("### 👤 User Profile")
    
    # Profile picture (using emoji as placeholder)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🧑‍💼")
    
    # User information
    st.markdown("""
    <div class="user-info">
        <div class="user-name">🙋‍♂️ Cikal Merdeka</div>
        <div class="user-email">📧 mcikalmerdeka@gmail.com</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chat statistics
    if "user_question_history" in st.session_state:
        total_questions = len(st.session_state["user_question_history"])
        st.metric("📊 Total Questions", total_questions)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", key="clear_chat"):
        st.session_state["user_question_history"] = []
        st.session_state["chat_answer_history"] = []
        st.session_state["chat_history"] = []
        st.rerun()
    
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **LangChain Documentation Helper**
        
        This AI-powered assistant helps you find information from the LangChain documentation quickly and accurately.
        
        🚀 **Features:**
        - Real-time documentation search
        - Contextual answers
        - Source citations
        - Chat history
        """)

# Main content area
st.markdown("<h1 class='main-header'>🦜 LangChain Documentation Helper</h1>", unsafe_allow_html=True)

# Initialize session states
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
    sources_string = "\n📚 **Sources:**\n"
    for i, source in enumerate(sources_list):
        sources_string += f"• {source}\n"
    return sources_string

# Chat interface
col1, col2 = st.columns([6, 1])

with col1:
    prompt = st.text_input(
        "Ask a question about LangChain:", 
        placeholder="e.g., How do I create a simple chain in LangChain?",
        key="user_input",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send 📤", key="send_btn", use_container_width=True)

# Process input when either button is clicked or Enter is pressed
if prompt and (send_button or prompt):
    with st.spinner("🔍 Searching documentation..."):
        # Add a small delay for better UX
        time.sleep(0.5)
        
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        
        source_documents = [doc.metadata["source"] for doc in generated_response["source_documents"]]
        
        # Format the response with the sources
        formatted_response = (
            f"{generated_response['result']}\n\n{create_sources_string(source_documents)}"
        )
        
        # Append to session state
        st.session_state["user_question_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))
        
        # Rerun to refresh the page (this will clear the input naturally)
        st.rerun()

# Display chat history
if st.session_state["user_question_history"] and st.session_state["chat_answer_history"]:
    st.markdown("### 💬 Chat History")
    
    # Create a container for the chat
    chat_container = st.container()
    
    with chat_container:
        # Reverse the order to show newest messages first
        for i, (gen_question, gen_answer) in enumerate(zip(
            reversed(st.session_state["user_question_history"]), 
            reversed(st.session_state["chat_answer_history"])
        )):
            # Create columns for better message layout
            with st.container():
                # User message
                with st.chat_message("user", avatar="🙋‍♂️"):
                    st.write(gen_question)
                
                # Assistant message
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(gen_answer)
                
                # Add a subtle separator between conversations
                if i < len(st.session_state["user_question_history"]) - 1:
                    st.markdown("---")

else:
    # Welcome message when no chat history exists
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #666;">
        <h3>👋 Welcome to LangChain Documentation Helper!</h3>
        <p>Ask me anything about LangChain documentation and I'll help you find the answers.</p>
        <p><strong>Try asking:</strong></p>
        <ul style="text-align: left; display: inline-block;">
            <li>"How do I create a simple chain?"</li>
            <li>"What are the different types of memory in LangChain?"</li>
            <li>"How to use embeddings with vector stores?"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Built using Streamlit and LangChain | 
    Made for Cikal Merdeka
</div>
""", unsafe_allow_html=True)