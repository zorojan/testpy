import streamlit as st
import pandas as pd
import os
from matching import perform_header_matching_and_grouping # Import the matching function

# Set page configuration
st.set_page_config(layout="wide")

# --- Step 1: Data Upload ---
st.sidebar.header("Step 1: Data Upload")

# Initialize session state for dataframes if not already present
if 'loaded_dataframes' not in st.session_state:
    st.session_state.loaded_dataframes = {}

# Add a flag to check if example data has been loaded
if 'example_data_loaded' not in st.session_state:
    st.session_state.example_data_loaded = False

# --- Load Example Data (run only once) ---
# Check if no dataframes are loaded AND example data hasn't been loaded yet
if not st.session_state.loaded_dataframes and not st.session_state.example_data_loaded:
    st.write("Loading example data...")
    try:
        # Assuming df1, df2, df3 DataFrames are defined in a previous cell and available
        st.session_state.loaded_dataframes['Source 1'] = df1
        st.session_state.loaded_dataframes['Source 2'] = df2
        st.session_state.loaded_dataframes['Source 3'] = df3
        st.session_state.example_data_loaded = True # Set flag to True after loading
        st.write("Example data loaded successfully.")
        # Rerun to update the UI with loaded data
        st.experimental_rerun()
    except NameError:
        st.warning("Example DataFrames (df1, df2, df3) not found. Please run the cell that defines them.")
    except Exception as e:
         st.error(f"An error occurred while loading example data: {e}")


uploaded_files = st.sidebar.file_uploader("Upload CSV, Excel, or JSON files", type=["csv", "xlsx", "xls", "json"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        # Check if the file is already loaded to avoid duplicates
        if file_name not in st.session_state.loaded_dataframes:
            try:
                if file_name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif file_name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif file_name.endswith('.json'):
                    df = pd.read_json(uploaded_file)

                st.session_state.loaded_dataframes[file_name] = df
                st.sidebar.success(f"Loaded {file_name}")
                # Clear example data flag if user uploads their own data
                st.session_state.example_data_loaded = False
                # Rerun to update the display with the new dataframe
                st.experimental_rerun()

            except Exception as e:
                st.sidebar.error(f"Error loading {file_name}: {e}")
        else:
            st.sidebar.info(f"{file_name} is already loaded.")

# Display loaded dataframes
if st.session_state.loaded_dataframes:
    st.sidebar.subheader("Loaded Dataframes")
    for file_name, df in st.session_state.loaded_dataframes.items():
        st.sidebar.write(f"- {file_name}")
        # Optional: Display head of each dataframe in the main area or an expander
        # with st.expander(f"Preview: {file_name}"):
        #      st.dataframe(df.head())
else:
     st.sidebar.info("No dataframes loaded yet.")

# --- Step 2: Column Header Matching and Grouping ---
st.sidebar.header("Step 2: Column Matching")

# Call the matching function from matching.py
# This function now contains the UI and logic for matching
perform_header_matching_and_grouping()


# --- Step 3: Data Merging (Placeholder) ---
st.sidebar.header("Step 3: Data Merging")
st.write("Data Merging functionality will be implemented here based on confirmed mappings.")
# This section will be developed later to use st.session_state.confirmed_column_mappings

# --- Step 4: Data Cleaning and Transformation (Placeholder) ---
st.sidebar.header("Step 4: Cleaning & Transformation")
st.write("Data Cleaning and Transformation functionality will be implemented here.")
# This section will be developed later

# --- Step 5: Export Merged Data (Placeholder) ---
st.sidebar.header("Step 5: Export Data")
st.write("Export merged and cleaned data functionality will be implemented here.")
# This section will be developed later
