import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(layout="wide")

@st.cache_resource
def load_data(file_path):
    script_dir = os.path.dirname(__file__)
    absolute_file_path = os.path.join(script_dir, file_path)

    if not os.path.exists(absolute_file_path):
        raise Exception(f"File not found: {absolute_file_path}")

    try:
        with open(absolute_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        st.error("Invalid JSON file.")
        return []

    return data

def style_dataframe(df):
    return df.style.set_properties(**{
        'font-size': '8pt',
        'font-weight': 'normal'
    })

def page():
    data = load_data('Library.json')
    df = pd.DataFrame(data)

    # Convert 'Creation Date' column to string
    df['CreationDate'] = df['CreationDate'].astype(str)

    df = df[['id', 'Link', 'Title', 'CreationDate', 'Author', 'Subject', 'Keywords']]
    df.columns = ['ID', 'Link', 'Title', 'Creation Date', 'Author', 'Subject', 'Keywords']

    # Sort by 'ID'
    df = df.sort_values('ID', ascending=True)

    st.title("Document Library List")

    # Create a new ID column with links
    df['ID'] = df.apply(lambda row: f'<a href="{row["Link"]}" target="_blank">{row["ID"]}</a>', axis=1)

    # Drop the 'Link' column as it's no longer needed
    df = df.drop(columns=['Link'])

    # Use st.markdown with unsafe_allow_html=True to display the dataframe with links
    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

# streamlit run Library.py

if __name__ == "__main__":
    page()



# streamlit run Library.py