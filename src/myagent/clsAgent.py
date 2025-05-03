import os
import asyncio
import streamlit as st
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, set_default_openai_api, set_tracing_disabled
from dotenv import load_dotenv
from asgiref.sync import async_to_sync

load_dotenv()

# Setup Interface
st.title("🤖 myAgent")
st.write(f"Developed by Rabeel Akram")
    
# Selectable Models
MODEL_OPTIONS = {
    "Gemini": {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash"
    },
    "Groq": {
        "api_key": os.getenv("GROQ_API_KEY"),
        "base_url": "https://api.groq.com/openai/v1/",
        "model": "llama3-70b-8192"
    },
    "Together AI": {
        "api_key": os.getenv("TOGETHER_API_KEY"),
        "base_url": "https://api.together.xyz/v1/",
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1"
    }
}

# Get the selected model (Use the first one as default)
selected_model = st.selectbox("Select Model", list(MODEL_OPTIONS.keys()))

# Get the Config (API key and base URL) for the selected model
model_config = MODEL_OPTIONS[selected_model]

# Check if API key and base URL are provided
if not model_config["api_key"]:
    st.error(f"API key not found for {selected_model}.")
    st.stop()
elif not model_config["base_url"]:
    st.error(f"Base URL not found for {selected_model}.")
    st.stop()

ai_client = AsyncOpenAI(
    api_key=model_config["api_key"],
    base_url=model_config["base_url"],
)

set_default_openai_api(ai_client)
set_tracing_disabled(True)

ai_model = OpenAIChatCompletionsModel(
    model=model_config["model"],
    openai_client=ai_client
)

def start():
    agent = Agent(name="Assistant", 
                  instructions="You are a helpful assistant", 
                  model=ai_model
    )

    
    st.subheader(f"{selected_model} ({model_config['model']})")
    
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