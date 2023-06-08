import streamlit as st
from auth0 import Auth0
from session import _get_state
import time
import os
import textwrap
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from elevenlabs import generate, play
from dotenv import load_dotenv

# Set Streamlit page config (must be the first Streamlit command)
st.set_page_config(layout="wide", page_title="FVA", page_icon=":speech_balloon:", initial_sidebar_state="expanded")

# Create an instance of Auth0
auth0 = Auth0()

# Get the session state
session_state = _get_state()

# Handle the authentication
if not session_state.authenticated:
    session_state.authenticated = auth0.login()

# Check if the user is authenticated
if session_state.authenticated:
    # Variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    api_key = os.getenv("PINECONE_API_KEY")
    env = os.getenv("PINECONE_ENVIRONMENT")
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    index_name = "yrtsinim"
    namespace = "ministry"
    text_field = 'text'

    # Auth0 config
    AUTH0_DOMAIN = os.environ.get('AUTH0_DOMAIN')
    AUTH0_CLIENT_ID = os.environ.get('AUTH0_CLIENT_ID')
    AUTH0_CLIENT_SECRET = os.environ.get('AUTH0_CLIENT_SECRET')
    AUTH0_CALLBACK_URL = os.environ.get('AUTH0_CALLBACK_URL')
    AUTH0_LOGOUT_CALLBACK_URL = os.environ.get('AUTH0_LOGOUT_CALLBACK_URL')

    # Vector Store
    pinecone.init(api_key=api_key, environment=env)
    index = pinecone.Index(index_name)
    embedding = OpenAIEmbeddings()
    vectorstore = Pinecone(index, embedding.embed_query, text_field)

    # VA Setup
    turbo_llm = ChatOpenAI(temperature=0.0, model_name="gpt-4")  # gpt-4 or gpt-3.5-turbo
    docsearch = Pinecone.from_existing_index(index_name=index_name, namespace=namespace, embedding=embedding)
    qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

    # Functions to format the response
    def wrap_text_preserve_newlines(text, width=110):
        lines = text.split('\n')
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
        wrapped_text = '\n'.join(wrapped_lines)
        return wrapped_text

    def process_llm_response(llm_response, show_sources=False, show_audio=False):
        response_text = wrap_text_preserve_newlines(llm_response['result'])
        st.write(response_text)

        if show_audio:
            audio = generate(text=response_text, voice="Bella", model="eleven_monolingual_v1", api_key=elevenlabs_api_key) # Bella Rachel Adam Josh
            play(audio)

        if show_sources:
            st.write('\n\nSources:')
            for source in llm_response["source_documents"]:
                page_content = source.page_content
                wrapped_text = wrap_text_preserve_newlines(page_content)
                st.write(wrapped_text)

    # Sidebar
    st.sidebar.title('Navigation')
    st.sidebar.markdown('## How to use')
    st.sidebar.text



# streamlit run app.py

