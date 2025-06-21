import streamlit as st
import pandas as pd
from pint import UnitRegistry
from datetime import datetime
import re # Keep re just in case for standardization, but remove value/unit match logic

# Initialize Pint UnitRegistry
ureg = UnitRegistry()

# Removed Value and Unit Matching Functions (extract_value_unit, value_unit_match)
# as per user's request. This module focuses on merging based on header groups
# and standardization of units/dates.


# Function to merge dataframes based on grouped headers
def merge_dataframes_by_grouped_headers(loaded_dataframes, grouped_headers):
    """
    Merges multiple dataframes based on grouped headers.

    Args:
        loaded_dataframes (dict): Dictionary of loaded dataframes.
        grouped_headers (list): A list of lists, where each inner list contains
                                 header info ({'header': col_name, 'source': df_name})
                                 for a group of similar headers.

    Returns:
        pd.DataFrame: Merged DataFrame with columns consolidated based on grouped headers.
    """
    if not loaded_dataframes or not grouped_headers:
        return pd.DataFrame() # Return empty DataFrame if no data or groups

    # Create a new DataFrame for the merged results
    merged_df = pd.DataFrame()

    # Iterate through each group of similar headers
    for group_index, group in enumerate(grouped_headers):
        if not group:
            continue # Skip empty groups

        # Determine the name for the consolidated column (e.g., use the first header in the group)
        consolidated_col_name = group[0]['header']
        # Make column name unique if necessary (e.g., add group index if headers are identical)
        # For simplicity, let's just use the first header name for now.

        # Collect data for this consolidated column from all source dataframes in the group
        group_data = pd.Series([], dtype='object') # Initialize an empty Series

        # Let's refine the merging logic: Create a temporary DataFrame with columns from this group.
        # Then, for each row, pick a value based on some rule (e.g., first non-null).

        temp_group_df = pd.DataFrame()
        for h_info in group:
             src_df_name = h_info['source']
             orig_h = h_info['header']
             if src_df_name in loaded_dataframes and orig_h in loaded_dataframes[src_df_name].columns:
                  # Rename the column to include source for clarity before consolidation
                  temp_group_df[f'{src_df_name}_{orig_h}'] = loaded_dataframes[src_df_name][orig_h]

        # Consolidate columns in the temporary DataFrame
        if not temp_group_df.empty:
             # Simple consolidation: fill NaN with values from other columns in the temp group
             # This assumes row alignment by index. If row alignment is needed by a key,
             # a different merging strategy is required (e.g., pd.merge based on a key column).
             # For now, let's stick to consolidating based on index for the same row across sources.
             consolidated_series = temp_group_df.iloc[:, 0] # Start with the first column
             for col_idx in range(1, temp_group_df.shape[1]):
                 consolidated_series = consolidated_series.fillna(temp_group_df.iloc[:, col_idx])

             merged_df[consolidated_col_name] = consolidated_series

    # After processing all groups, add any columns from original dataframes that were NOT grouped
    all_grouped_original_headers = set()
    for group in grouped_headers:
        for header_info in group:
             all_grouped_original_headers.add((header_info['source'], header_info['header'])) # Store as (source, header) tuple

    for df_name, df in loaded_dataframes.items():
        for col in df.columns:
            if (df_name, col) not in all_grouped_original_headers:
                 # Add this ungrouped column, potentially with source prefix
                 unique_col_name = f'{df_name}_{col}'
                 if unique_col_name not in merged_df.columns: # Avoid duplicates if column names are identical across files but not grouped
                    merged_df[unique_col_name] = df[col]
                 else:
                     # Handle cases where ungrouped columns have same name - potentially rename or raise warning
                     pass # For now, assume unique names or they would have been grouped


    return merged_df


# Function to standardize units
def standardize_units(df):
     """
     Identifies potential quantity columns and prompts user for target units,
     then standardizes the units.
     """
     st.subheader("Unit Standardization")
     st.write("Specify target units for potential quantity columns.")

     # Identify potential quantity columns (simplified: look for 'price', 'qty', 'quantity')
     potential_quantity_cols = [col for col in df.columns if 'price' in col.lower() or 'qty' in col.lower() or 'quantity' in col.lower()]

     quantity_cols_info = {}
     if potential_quantity_cols:
         for col in potential_quantity_cols:
              target_unit = st.text_input(f"Target unit for '{col}' (e.g., 'usd', 'kg', 'each'):", key=f"unit_input_{col}")
              if target_unit:
                  try:
                      ureg(target_unit) # Validate if unit is parseable by pint
                      quantity_cols_info[col] = target_unit
                  except Exception:
                      st.warning(f"Invalid unit '{target_unit}' for column '{col}'. Please enter a valid unit recognized by Pint.")


     if quantity_cols_info:
         standardized_df = df.copy() # Work on a copy
         for col, target_unit_str in quantity_cols_info.items():
             if col in standardized_df.columns:
                 try:
                     target_unit = ureg(target_unit_str)
                     standardized_values = []
                     for value in standardized_df[col]:
                         if pd.notna(value):
                             try:
                                 # Attempt to convert the value (assuming it might have a unit or is a number)
                                 quantity = ureg(str(value)) # Try parsing as a quantity string
                                 standardized_values.append(quantity.to(target_unit).magnitude)
                             except Exception:
                                 # If parsing as quantity string fails, try treating as a number
                                 try:
                                     num_value = float(value)
                                     # If it's just a number, assume it's already in the target unit
                                     standardized_values.append(num_value)
                                 except (ValueError, TypeError):
                                     standardized_values.append(None) # Cannot standardize
                         else:
                             standardized_values.append(None) # Keep NaN

                     standardized_df[f'{col}_standardized_{target_unit_str}'] = standardized_values # Add a new column with standardized values

                 except Exception as e:
                     st.warning(f"Could not standardize unit for column '{col}' to '{target_unit_str}': {e}")
         return standardized_df
     else:
          st.info("No target units specified for unit standardization.")
          return df # Return original df if no standardization performed


# Function to standardize dates
def standardize_dates(df, target_format='%Y-%m-%d'):
    """
    Identifies potential date columns and standardizes their format.
    """
    st.subheader("Date Standardization")

    # Identify potential date columns (simplified: look for 'date')
    date_cols = [col for col in df.columns if 'date' in col.lower()]

    if date_cols:
        target_date_format = st.text_input("Target date format (e.g., '%Y-%m-%d'):", value='%Y-%m-%d', key="date_format_input")
        standardized_df = df.copy() # Work on a copy
        for col in date_cols:
            if col in standardized_df.columns:
                standardized_dates = []
                for date_value in standardized_df[col]:
                    if pd.notna(date_value):
                        try:
                            # Try converting to datetime object first
                            if isinstance(date_value, pd.Timestamp):
                                standardized_dates.append(date_value.strftime(target_format))
                            elif isinstance(date_value, str):
                                # Attempt to parse string dates with common formats
                                try:
                                    dt_obj = pd.to_datetime(date_value)
                                    standardized_dates.append(dt_obj.strftime(target_format))
                                except Exception:
                                     standardized_dates.append(str(date_value)) # Keep original if parsing fails
                                else:
                                     standardized_dates.append(str(date_value)) # Keep original for other types
                        except Exception:
                            standardized_dates.append(str(date_value)) # Fallback to original string representation
                    else:
                        standardized_dates.append(None) # Keep NaN

                standardized_df[f'{col}_standardized'] = standardized_dates # Add a new column with standardized dates
        return standardized_df
    else:
        st.info("No date columns identified for date standardization.")
        return df # Return original df if no standardization performed


def perform_standardization_and_merging(loaded_dataframes):
    """
    Provides UI for reviewing and confirming grouped headers, performs merging
    based on confirmed groups, and then performs data standardization.

    Args:
        loaded_dataframes (dict): Dictionary of loaded dataframes.

    Returns:
        pd.DataFrame or None: The final standardized and merged DataFrame, or None.
    """
    st.header("Data Standardization and Merging")
    standardized_and_merged_df = None # Initialize to None

    # Check if grouped header results exist in session state
    if 'grouped_header_matches' in st.session_state and st.session_state.grouped_header_matches:
        st.subheader("Review and Confirm Header Groups for Merging")
        st.write("Review the suggested header groups. Headers within the same group will be merged into a single column.")

        grouped_headers = st.session_state.grouped_header_matches

        # Display grouped headers for review
        st.subheader("Suggested Header Groups")
        # Prepare data for display
        display_data = []
        for group_index, group in enumerate(grouped_headers):
            for item in group:
                display_data.append({
                    'Group ID': group_index + 1,
                    'Header Name': item['header'],
                    'Source File': item['source']
                })
        grouped_headers_df = pd.DataFrame(display_data)
        st.dataframe(grouped_headers_df)

        # Simple confirmation button for now. More advanced could allow editing groups.
        confirm_groups = st.button("Confirm Header Groups and Perform Merging", key="confirm_header_groups_button")

        if confirm_groups:
             st.write("Performing merging based on confirmed header groups...")

             # Perform merging based on the confirmed header groups
             # The merge_dataframes_by_grouped_headers function handles merging multiple DFs
             merged_df = merge_dataframes_by_grouped_headers(
                 loaded_dataframes, # Pass all loaded dataframes
                 grouped_headers    # Pass the grouped headers
             )

             if not merged_df.empty:
                 st.subheader("Merged Data (Before Standardization)")
                 st.dataframe(merged_df.head())

                 # --- Standardization Steps on the Merged Data ---

                 # 1. Unit Standardization
                 df_after_unit_standardization = standardize_units(merged_df)
                 # The standardize_units function now handles its own UI and returns the modified df

                 # 2. Date Standardization
                 df_after_date_standardization = standardize_dates(df_after_unit_standardization)
                 # The standardize_dates function now handles its own UI and returns the modified df

                 standardized_and_merged_df = df_after_date_standardization # Final result after all standardizations

                 st.subheader("Final Standardized and Merged DataFrame")
                 st.dataframe(standardized_and_merged_df)
                 st.session_state.standardized_dataframe = standardized_and_merged_df # Store in session state

                 # Allow downloading the final results
                 csv_output_final = standardized_and_merged_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download Final Merged Results as CSV",
                     data=csv_output_final,
                     file_name=f"standardized_merged_results.csv", # Generic name as multiple files are merged
                     mime="text/csv",
                     key=f"download_standardized_merged_results_grouped"
                 )

             else:
                 st.warning("Merging resulted in an empty DataFrame. Please check the header groups.")

    elif 'loaded_dataframes' in st.session_state and len(st.session_state.loaded_dataframes) >= 1: # Can group headers even with one file
         st.info("Please perform column header matching and grouping in Step 2 to generate results for merging.")
    else:
         st.info("Please upload at least one price list file and perform header matching and grouping to proceed with merging and standardization.")


    return standardized_and_merged_df
