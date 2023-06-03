import streamlit as st
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

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="Chat App", page_icon=":speech_balloon:", initial_sidebar_state="expanded")

# Load Pinecone
load_dotenv()

# Variables
openai_api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
index_name = "yrtsinim"
namespace = "ministry"
text_field = 'text'

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

# Streamlit App
st.title('Forestry Virtual Assistant')

query = st.text_input('Enter Your Question:')
show_sources = st.checkbox('Show Sources')

# Model selection
model_name = st.selectbox('Select model:', ('gpt-4', 'gpt-3.5-turbo'))

if st.button('Submit'):
    # VA Setup
    turbo_llm = ChatOpenAI(temperature=0.0, model_name=model_name)
    docsearch = Pinecone.from_existing_index(index_name=index_name, namespace=namespace, embedding=embedding)
    qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

    start_time = time.time()
    llm_response = qa_chain({"query": query})
    response_text = wrap_text_preserve_newlines(llm_response['result'])
    st.write(response_text)

    if show_sources:
        st.write('\n\nSources:')
        for i, source in enumerate(llm_response["source_documents"]):
            with st.expander(f'Source {i+1}'):
                page_content = source.page_content
                wrapped_text = wrap_text_preserve_newlines(page_content)
                st.write(wrapped_text)

    end_time = time.time()
    total_time = end_time - start_time
    st.write("\nTotal runtime: {:.2f} seconds".format(total_time))
    st.write(f"Model used: {model_name}")




# streamlit run app.py

