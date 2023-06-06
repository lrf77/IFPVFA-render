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

# Set Streamlit page config (must be the first Streamlit command)
st.set_page_config(layout="wide", page_title="FVA", page_icon=":speech_balloon:", initial_sidebar_state="expanded")

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

# Sidebar
st.sidebar.title('Navigation')
st.sidebar.markdown('## How to use')
st.sidebar.markdown('<p style="font-size:10px">1. Enter your question in the text input field in the main area of the app.</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:10px">2. If you want to see the sources that the answer is based on, check the "Show Sources" checkbox.</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:10px">3. Select the model you want to use from the dropdown menu. The options are "gpt-4" and "gpt-3.5-turbo".</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:10px">4. Click the "Submit" button to get the answer to your question.</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:10px">5. The answer will appear in the main area of the app. If you checked the "Show Sources" checkbox, the sources will appear below the answer.</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:10px">6. If you want to ask another question, simply enter it in the text input field and click "Submit" again.</p>', unsafe_allow_html=True)
st.sidebar.markdown('## About')
st.sidebar.markdown('<p style="font-size:10px">This is the Forestry Virtual Assistant - FVA, a tool designed to provide quick and accurate answers to your forestry-related questions. The FVA uses advanced AI models to search through a comprehensive library of forestry documents and provide responses based on the most relevant information. The library includes a wide range of documents from the BC Ministry of Forests, Timber Pricing Branch, Timber Supply Review, and more. Whether you are looking for specific information or general knowledge, the FVA is here to assist you.</p>', unsafe_allow_html=True)
st.sidebar.markdown('## FAQ')
st.sidebar.markdown('<p style="font-size:10px"><b>Q1: How does the Forestry Virtual Assistant work?</b><br>A1: The FVA uses AI models to understand your question and search through a library of forestry documents for the most relevant information. It then presents this information as a clear and concise answer.</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:10px"><b>Q2: What kind of questions can I ask?</b><br>A2: You can ask any question related to forestry. The FVA is designed to handle a wide range of topics, from specific details about forestry practices to general information about the forestry sector.</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:10px"><b>Q3: How accurate are the answers?</b><br>A3: The FVA strives to provide the most accurate information possible. However, as with any AI tool, the accuracy can vary depending on the complexity of the question and the information available in the library documents.</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:10px"><b>Q4: Can I see the sources of the information?</b><br>A4: Yes, you can choose to see the sources of the information by checking the "Show Sources" checkbox before submitting your question.</p>', unsafe_allow_html=True)


# List of documents
documents = [
    "BC Ministry of Forests - BRIDGE STANDARDS MANUAL - Mar. 7, 2023",
    "BC Ministry of Forests - GROWTH AND YIELD PREDICTION SYSTEMS - February 1991",
    "BC Timber Sales Annual Performance Report 2018 - 2019",
    "BC Timber Sales Annual Performance Report 2019 - 2020",
    "BCTS BC Timber Sales - ANNUAL PERFORMANCE REPORT - APRIL 1, 2020 – MARCH 31, 2021",
    "BCTS BC Timber Sales - ANNUAL PERFORMANCE REPORT - APRIL 1, 2021 – MARCH 31, 2022",
    "BCTS BC Timber Sales - BUSINESS PLAN - 2022/23 – 2024/25",
    "BCTS BC Timber Sales - Climate Change Action Strategy - February 2, 2015",
    "BCTS BC Timber Sales - Environmental Stewardship and Sustainability Report - April 2021",
    "BCTS BC Timber Sales - Goals, Objectives, and Principles",
    "BCTS BC Timber Sales - QUARTER PERFORMANCE REPORT - APRIL 1, 2022 – JUNE 30, 2022",
    "BCTS BC Timber Sales - QUARTER PERFORMANCE REPORT - APRIL 1, 2022 – September 30, 2022",
    "BCTS BC Timber Sales - QUARTER PERFORMANCE REPORT - APRIL 1, 2022 – December 31, 2022",
    "BILL 21 – 2019 - FOREST AND RANGE PRACTICES - AMENDMENT ACT, 2019",
    "BILL 23 - Forests Statutes Amendment Act - 2021",
    "Boundary Timber Supply Area - Rationale for Allowable Annual Cut (AAC) Determination - Effective May 22, 2014",
    "Boundary Timber Supply Area - Timber Supply Review - Data Package - June 2011",
    "Boundary TSA - Timber Supply Analysis - Public Discussion Paper",
    "Cascadia - Timber Supply Area - Rationale for Allowable Annnual Cut (AAC) - Determination - Effective January 23, 2020",
    "CSV File Description of Common Data Fields for Each CSV File",
    "Cutting Permit and Road Tenure Administration Manual - September 2020",
    "Forest & Range Evaluation Program - THREE-YEAR STRATEGIC PLAN - 2020/21 to 2022/23 - August 10, 2020",
    "Forest Analysis & Inventory Branch - Provincial Guide for the preparation of Information Packages and Analysis Reports for Area-based Tenures - June 2021",
    "Forest Analysis and Inventory Branch - Timber Supply Review - Timber Supply Areas - November 2016",
    "Forest Analysis and Inventory Branch - Timber Supply Review - Tree Farm Licences - November 2016",
    "Forest and Range Practices Act - Administrative Guide for Forest Stewardship Plans (FSPs) Volume I Preparation and Approval of an FSP - Version 2.1 August 2009",
    "Forest and Range Practices Act - Administrative Guide for Forest Stewardship Plans (FSPs) Volume II Operating Under an Approved FSP - Version 1.1a March 2010",
    "FOREST HEALTH UNIT, RESOURCE PRACTICES BRANCH - 2019-2022 Provincial Forest Health Strategy",
    "Forest Inventory Strategic Plan - February 2013",
    "FOREST PRACTICES CODE OF BRITISH COLUMBIA ACT - PDF Version [Pre-Jan. 31, 2004 amendments]",
    "Forest Science, Planning, and Practices Branch - Silviculture Survey Procedures Manual - April 1, 2023",
    "FOREST TENURES BRANCH - Licence to Cut Administration Manual - Version 3.1 – April 2, 2020",
    "Government Intentions to Modernize Forest Policy - Emngagement Grouping 1 topic overview",
    "Government Intentions to Modernize Forest Policy - Emngagement Grouping 2 topic overview",
    "Impacts of 2017 Fires on Timber Supply in the Cariboo Region - February 2018",
    "Impacts of 2018 Fires on Forests and Timber Supply in British Columbia - April 2019",
    "Impacts of 2021 Fires on Forests and Timber Supply in British Columbia - April 2022",
    "Interior Forest Sector Renewal Policy and Program Engagement Discussion Paper - Summer 2019",
    "Kamloops Timber Supply Area - Rationale for Allowable Annual Cut (AAC) Determination - Effective May 5, 2016",
    "Kamloops Timber Supply Area - Timber Supply Analysis - Discussion Paper - September 2015",
    "Kamloops Timber Supply Area - Timber Supply Review - Data Package - UPDATE September 2015",
    "Kootenay Lake Timber Supply Area - Timber Supply Analysis - Discussion Paper - May 2023",
    "Kootenay Lake Timber Supply Area - Timber Supply Review - Data Package - Novemeber 2020",
    "Kootenay Lake TSA - Timber Supply Analysis Discussion Paper - September 2009",
    "Kootenay Lake TSA - Timber Supply Information Report - December 2008",
    "Kootenay Lake TSA - Timber Supply Rationale for Allowable Annual Cut (AAC) Determination - Effective August 12, 2010",
    "Kootenay Lake TSA - Timber Supply Review Data Package - July 2008",
    "Ministry of Forests, Lands, and Natural Resource Operations - Apportionment System - Linkages and Licences",
    "Modernizing Forest Policy in British Columbia: Setting The Intention and Leading the Forest Sector Transition",
    "North and South Area Timber Pricing - Interior Engineering Cost Estimate Procedures - Effective: July 1, 2022",
    "Okanagan - Timber Supply Area - Timber Supply Analysis - Discussion Paper - January 2021",
    "Okanagan Timber Supply Area - Timber Supply Analysis - Discussion Paper - January 2021",
    "Okanagan Timber Supply Area - Timber Supply Review - Data Package - December 2017",
    "Partial Cutting - Ministry key training materials, standards, systems, and growth and yield models",
    "Planting Quality Inspection - Guide to Completing the FS 704 - Effective April 2012",
    "Silvicultural Systems Handbook for British Columbia - March 2003",
    "Special Investigation: Conserving Fish Habitat under the Forest and Range Practices Act - PART 2: An Evaluation of Forest and Range Practices on the Ground - May 2020",
    "Timber Pricing Branch - Concurrent Residual Harvest System – Interior (CRHS) - June 2020",
    "TIMBER PRICING BRANCH - Interior Appraisal Manual - Effective July 1, 2022 - Cost Base of: 2020",
    "TIMBER PRICING BRANCH - Interior Appraisal Manual - Effective July 1, 2022 - Cost Base of: 2020 - Amendment No. 1,2,3 - May 1, 2023",
    "Timber Pricing Branch - Interior MARKET PRICING SYSTEM - Update – July 1, 2022",
    "Timber Pricing Branch - Interior Timber Pricing Training Modules 1 – 5 - July 2016",
    "Timber Pricing Branch - Scaling Manual - Effective November 1, 2011 - Includes Amendments No. 1 to No. 5 - Latest Effective Date November 15, 2021",
    "Timber Pricing Branch - Waste Assessments",
    "Timber Supply Review - Analysis Report – Cascadia TSA - Version 1.5 Draft",
    "Timber Supply Review - Information Package – Cascadia TSA - Version 1.61",
    "Timber Supply Review (TSR) Document Descriptions",
    "Timber Supply Review Backgrounder",
    "UBC Forestry Handbook for British Columbia - Fifth Edition - PDF Part 1",
    "UBC Forestry Handbook for British Columbia - Fifth Edition - PDF Part 2",
    "What we Heard: Engagement on Interior Forest Sector Renewal - February 2020",
    "A NEW FUTURE FOR OLD FORESTS: A Strategic Review of How British Columbia Manages for Old Forests Within its Ancient Ecosystems",
    "A Writing Guidebook for the Natural Sciences - David Godsall - University of British Columbia - Faculty of Forestry"
]

# Create an expander for the documents in the sidebar
with st.sidebar.expander("Library Documents"):
    for document in documents:
        st.markdown(f'<p style="font-size:9px">{document}</p>', unsafe_allow_html=True)

# Main area
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


# streamlit run appcopy.py

