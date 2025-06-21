import streamlit as st
import pandas as pd
import os

def load_data():
    """
    Allows users to upload price list files and loads them into Pandas DataFrames.
    Updates st.session_state.loaded_dataframes.
    """
    st.header("Data Ingestion")
    st.write('Upload your price lists to start the process.')

    uploaded_files = st.file_uploader(
        "Choose price list files (CSV or XLSX)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        # Process only new files or if the uploaded files list has changed
        current_file_names = {f.name for f in uploaded_files}
        loaded_file_names = set(st.session_state.loaded_dataframes.keys())

        if current_file_names != loaded_file_names:
             st.session_state.loaded_dataframes = {} # Clear old dataframes if files change

             for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                st.write(f"Processing file: {file_name}")

                try:
                    if file_name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif file_name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                    else:
                        st.error(f"Unsupported file type for {file_name}")
                        continue

                    st.session_state.loaded_dataframes[file_name] = df
                    st.success(f"Successfully loaded {file_name}")
                    # Optionally display the first few rows
                    st.dataframe(df.head())

                except Exception as e:
                    st.error(f"Error loading file {file_name}: {e}")

    # This function no longer returns the dict, it updates session state directly.
    # Returning None or just letting the function end is appropriate.
    # For clarity, we can explicitly return None.
    # If we need to signal that new data was loaded, we could return a boolean,
    # but app.py can check the state of st.session_state.loaded_dataframes.
    return None
