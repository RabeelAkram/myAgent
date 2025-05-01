import os
import asyncio
import streamlit as st
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, set_default_openai_api, set_tracing_disabled
from dotenv import load_dotenv
from asgiref.sync import async_to_sync

load_dotenv()

# Load environment variables from .env file
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini_model_name = "gemini-2.0-flash"

# Check if the environment variables are set
if not gemini_api_key or len(gemini_api_key) <= 0:
    st.error("Gemini API Key Not Found.")
    st.stop()

gemini_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url=gemini_base_url
)

set_default_openai_api(gemini_client)
set_tracing_disabled(True)

gemini_model = OpenAIChatCompletionsModel(
    model=gemini_model_name,
    openai_client=gemini_client
)

def start():
    agent = Agent(name="Assistant", 
                  instructions="You are a helpful assistant", 
                  model=gemini_model
    )

    
    
    # Setup Interface
    st.title("🤖 myAgent")
    st.subheader(f"Developed by Rabeel Akram")
    st.text(f"This AI agent is powered by {gemini_model_name}")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    
    prompt = st.chat_input("Ask Anything...")

    # Get response with both button and enter key
    if prompt:
        # Display user's message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Create agent and run
        with st.spinner("Thinking..."):
            # Run the agent with the provided prompt
            response = async_to_sync(Runner.run)(agent, prompt)
            ai_reply = response.final_output

        st.chat_message("assistant").markdown(ai_reply)
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})

# Render the UI
start()