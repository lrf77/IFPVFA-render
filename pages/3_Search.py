# Import the necessary modules
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import DuckDuckGoSearchRun


# Set Streamlit page config (must be the first Streamlit command)
st.set_page_config(layout="wide", page_title="FVA", page_icon=":evergreen_tree:")

# Load Pinecone
load_dotenv()

# Load OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("LangChain - Chat with Search")

# """
# In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
# Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
# """

import streamlit as st

# Add a button in the sidebar
if st.sidebar.button('Clear Chat'):
    # Clear the messages when the button is clicked
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web using LangChain's DuckDuckGo Search tool. How can I help you?"}
    ]

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web using LangChain's DuckDuckGo Search tool. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Ask me a question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in search_agent.run(input=st.session_state.messages, callbacks=[StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)], stream=True):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# streamlit run 3_Search.py