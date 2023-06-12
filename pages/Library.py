import streamlit as st
import pandas as pd
import json
import os

@st.cache(allow_output_mutation=True)
def load_data(file_path):
    """
    Load data from a JSON file.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    list: The data from the JSON file.
    """
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(__file__)

    # Construct the absolute path of the file
    absolute_file_path = os.path.join(script_dir, file_path)

    if not os.path.exists(absolute_file_path):
        raise Exception(f"File not found: {absolute_file_path}")

    with open(absolute_file_path) as f:
        data = json.load(f)

    return data


def style_dataframe(df):
    """
    Style a DataFrame for display.

    Parameters:
    df (pandas.DataFrame): The DataFrame to style.

    Returns:
    pandas.io.formats.style.Styler: The styled DataFrame.
    """
    return df.style.set_properties(**{
        'font-size': '8pt',
        'font-weight': 'normal'
    })

def page():
    """
    The main function of the app.
    """
    # Load the data
    data = load_data('Library.json')

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Convert 'Creation Date' column to string
    df['/CreationDate'] = df['/CreationDate'].astype(str)

    # Select and rename columns
    df = df[['id', '/Title', '/CreationDate', '/Author', '/Subject', '/Keywords']]
    df.columns = ['ID', 'Title', 'Creation Date', 'Author', 'Subject', 'Keywords']

    # Add a title to the app
    st.title("MoF Document Library")

    # Display the DataFrame
    st.table(style_dataframe(df))

# streamlit run library.py