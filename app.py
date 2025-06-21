import streamlit as st
import pandas as pd
import os

# Import functions from the new module files
from ingestion import load_data
from matching import perform_header_matching_and_grouping
from merging import perform_merging
from standardization import perform_standardization

st.title('Price List Matching and Standardization')
st.write('Upload your price lists to start the process.')

# Initialize session state variables if not already present
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'loaded_dataframes' not in st.session_state:
    st.session_state.loaded_dataframes = {}
if 'grouped_header_matches' not in st.session_state:
    st.session_state.grouped_header_matches = []
if 'confirmed_column_mappings' not in st.session_state:
    st.session_state.confirmed_column_mappings = {} # Store confirmed mappings
if 'merged_dataframe' not in st.session_state:
    st.session_state.merged_dataframe = None
if 'standardized_dataframe' not in st.session_state:
    st.session_state.standardized_dataframe = None

# Define the total number of steps
TOTAL_STEPS = 5

# --- Wizard Steps ---

if st.session_state.step == 1:
    st.header("Step 1: Data Ingestion")
    # Call the load_data function from the ingestion module
    load_data() # This function updates st.session_state.loaded_dataframes

elif st.session_state.step == 2:
    st.header("Step 2: Column Header Matching and Grouping")
    if st.session_state.loaded_dataframes:
        # Call the perform_header_matching_and_grouping function
        perform_header_matching_and_grouping() # This function updates st.session_state.grouped_header_matches and confirmed_column_mappings
    else:
        st.info("Please go back to Step 1 to upload data.")


elif st.session_state.step == 3:
    st.header("Step 3: Data Merging")
    # Check if confirmed column mappings exist before displaying merging options
    if st.session_state.get('confirmed_column_mappings', {}): # Check if confirmed_column_mappings is not empty
        # Call the perform_merging function
        perform_merging() # This function updates st.session_state.merged_dataframe
    elif st.session_state.get('grouped_header_matches', []):
         st.info("Please select headers in Step 2 and confirm the mappings.")
    elif st.session_state.get('loaded_dataframes', {}):
        st.info("Please go back to Step 2 to perform Column Header Matching and Grouping.")
    else:
        st.info("Please go back to Step 1 to upload data.")


elif st.session_state.step == 4:
    st.header("Step 4: Data Standardization")
    # Check if a merged dataframe exists before displaying standardization options
    if st.session_state.merged_dataframe is not None:
        # Call the perform_standardization function
        # This function now accesses merged_dataframe from session state
        perform_standardization() # This function updates st.session_state.standardized_dataframe
    elif st.session_state.get('confirmed_column_mappings', {}):
         st.info("Please perform Data Merging in Step 3 first.")
    elif st.session_state.get('grouped_header_matches', []):
         st.info("Please go back to Step 2 and confirm header mappings, then perform merging in Step 3.")
    elif st.session_state.get('loaded_dataframes', {}):
         st.info("Please go back to Step 2 to perform Column Header Matching and Grouping.")
    else:
        st.info("Please go back to Step 1 to upload data.")


elif st.session_state.step == 5:
    st.header("Step 5: Final Result")
    # Check if standardized data exists to display the final result
    if st.session_state.standardized_dataframe is not None:
        st.write("Here is the final standardized and merged data.")
        st.dataframe(st.session_state.standardized_dataframe)

        # Allow downloading the final results
        csv_output_final = st.session_state.standardized_dataframe.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Final Standardized and Merged Results as CSV",
            data=csv_output_final,
            file_name=f"final_standardized_merged_results.csv",
            mime="text/csv",
            key=f"download_final_results"
        )
    else:
        st.info("Please complete all previous steps to see the final result.")


# --- Navigation Buttons ---
# Add a container for the buttons at the bottom
button_container = st.container()

with button_container:
    col1, col2 = st.columns(2)

    with col1:
        # Previous button logic
        if st.button("Previous", key="prev_button", disabled=(st.session_state.step <= 1)):
            st.session_state.step -= 1
            st.rerun() # Rerun the app to display the previous step

    with col2:
        # Next button logic
        # Disable 'Next' if current step's required data is not available
        disable_next = False
        if st.session_state.step == 1 and not st.session_state.loaded_dataframes:
            disable_next = True
        elif st.session_state.step == 2 and not st.session_state.get('confirmed_column_mappings', {}):
            disable_next = True
        elif st.session_state.step == 3 and st.session_state.merged_dataframe is None:
             disable_next = True
        elif st.session_state.step == 4 and st.session_state.standardized_dataframe is None:
             disable_next = True
        elif st.session_state.step == 5: # Already at the last step
             disable_next = True


        if st.button("Next", key="next_button", disabled=disable_next):
            st.session_state.step += 1
            st.rerun() # Rerun the app to display the next step
