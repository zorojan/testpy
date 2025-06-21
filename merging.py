import streamlit as st
import pandas as pd

# Function to merge dataframes based on confirmed column mappings
def merge_dataframes_by_confirmed_mappings(loaded_dataframes, confirmed_mappings):
    """
    Merges multiple dataframes based on user-confirmed column mappings.

    Args:
        loaded_dataframes (dict): Dictionary of loaded dataframes.
        confirmed_mappings (dict): A dictionary where keys are the desired
                                   consolidated column names, and values are
                                   dictionaries mapping source file names to
                                   the selected header name from that file.
                                   E.g., {'Consolidated_Weight': {'file1.csv': 'Weight (kg)', 'file2.xlsx': 'W'}}

    Returns:
        pd.DataFrame: Merged DataFrame with columns consolidated based on mappings.
    """
    if not loaded_dataframes or not confirmed_mappings:
        st.warning("No dataframes loaded or no confirmed column mappings provided for merging.")
        return pd.DataFrame() # Return empty DataFrame if no data or mappings

    # Create a new DataFrame for the merged results
    merged_df = pd.DataFrame()

    # Keep track of original columns that have been included in the merged_df
    included_original_cols = set() # Store as (source, header) tuple

    # Process each confirmed mapping (which defines a consolidated column)
    for consolidated_col_name, source_mappings in confirmed_mappings.items():
        # Create a temporary DataFrame with columns from this group based on confirmed selections
        temp_group_df = pd.DataFrame()
        for source_file, original_header in source_mappings.items():
            if source_file in loaded_dataframes and original_header in loaded_dataframes[source_file].columns:
                # Add the selected original column to the temporary DataFrame
                # Rename the column to include source for clarity before consolidation (optional but good practice)
                temp_group_df[f'{source_file}_{original_header}'] = loaded_dataframes[source_file][original_header]
                included_original_cols.add((source_file, original_header))

        # Consolidate columns in the temporary DataFrame
        if not temp_group_df.empty:
             # Simple consolidation: fill NaN with values from other columns in the temp group
             # This assumes row alignment by index. If row alignment is needed by a key,
             # a different merging strategy is required (e.g., pd.merge based on a key column).
             # For now, let's stick to consolidating based on index for the same row across sources.
             consolidated_series = temp_group_df.iloc[:, 0] # Start with the first column
             for col_idx in range(1, temp_group_df.shape[1]):
                 consolidated_series = consolidated_series.fillna(temp_group_df.iloc[:, col_idx])

             # Add the consolidated column to the final merged DataFrame
             # Ensure the consolidated column name is unique before adding (should be unique based on confirmed_mappings keys)
             if consolidated_col_name not in merged_df.columns:
                  merged_df[consolidated_col_name] = consolidated_series
             else:
                  st.warning(f"Consolidated column name '{consolidated_col_name}' is not unique. Skipping.")


    # Add any columns from original dataframes that were NOT included in the confirmed mappings
    for df_name, df in loaded_dataframes.items():
        for col in df.columns:
            if (df_name, col) not in included_original_cols:
                 # Add this ungrouped column, potentially with source prefix
                 unique_col_name = f'{df_name}_{col}'
                 # Simple handling for potential name conflicts if ungrouped columns
                 # from different sources have the same name. Append a counter.
                 original_unique_col_name = unique_col_name
                 counter = 1
                 while unique_col_name in merged_df.columns: # Check against already merged columns
                     unique_col_name = f"{original_unique_col_name}_{counter}"
                     counter += 1

                 merged_df[unique_col_name] = df[col]


    return merged_df


def perform_merging():
    """
    Provides UI for confirming column mappings (from session state),
    performs merging based on confirmed mappings, and updates st.session_state.merged_dataframe.
    """
    st.header("Data Merging")
    merged_df = None # Initialize to None

    # Access loaded dataframes and confirmed column mappings from session state
    loaded_dataframes = st.session_state.get('loaded_dataframes', {})
    confirmed_mappings = st.session_state.get('confirmed_column_mappings', {})


    # Check if confirmed mappings exist
    if confirmed_mappings:
        st.subheader("Confirmed Column Mappings")
        st.write("The following mappings will be used to merge columns:")

        # Display confirmed mappings for review
        # Prepare data for display
        display_data = []
        for consolidated_col, mappings in confirmed_mappings.items():
            display_data.append({
                'Consolidated Column Name': consolidated_col,
                'Source File Mappings': str(mappings) # Display mappings as string
            })
        confirmed_mappings_df = pd.DataFrame(display_data)
        st.dataframe(confirmed_mappings_df)


        # Button to trigger merging
        # Use a unique key for the button in this step
        if st.button("Perform Merging", key="perform_merging_button_step"):
             st.write("Performing merging based on confirmed column mappings...")

             # Perform merging based on the confirmed header groups
             # PASS the loaded_dataframes and confirmed_mappings to the helper function
             merged_df = merge_dataframes_by_confirmed_mappings(
                 loaded_dataframes, # Pass loaded dataframes
                 confirmed_mappings # Pass confirmed mappings
             )

             if not merged_df.empty:
                 st.subheader("Merged Data")
                 st.dataframe(merged_df.head())

                 # Store the merged dataframe in session state for standardization
                 st.session_state.merged_dataframe = merged_df

                 # Allow downloading the merged results (before standardization)
                 csv_output_merged = merged_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download Merged Data as CSV (Before Standardization)",
                     data=csv_output_merged,
                     file_name=f"merged_data_before_standardization.csv",
                     mime="text/csv",
                     key=f"download_merged_data_step_3"
                 )

             else:
                 st.warning("Merging resulted in an empty DataFrame. Please check the confirmed mappings.")

    elif st.session_state.get('grouped_header_matches', []):
        st.info("Please select headers in Step 2 and confirm the mappings.")
    elif st.session_state.get('loaded_dataframes', {}): # If data is loaded but not grouped/mapped
        st.info("Please go back to Step 2 to perform Column Header Matching and Grouping.")
    else: # If no data loaded
        st.info("Please go back to Step 1 to upload data.")


    # This function updates session state directly and returns None.
    return None
