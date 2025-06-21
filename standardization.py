import streamlit as st
import pandas as pd
from pint import UnitRegistry
from datetime import datetime
import re

# Initialize Pint UnitRegistry
ureg = UnitRegistry()

# Function to standardize units
def standardize_units(df):
     """
     Identifies potential quantity columns and prompts user for target units,
     then standardizes the units. Returns the dataframe with standardized columns.
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


     standardized_df = df.copy() # Work on a copy
     if quantity_cols_info:
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
     else:
          st.info("No target units specified for unit standardization.")

     return standardized_df


# Function to standardize dates
def standardize_dates(df, target_format='%Y-%m-%d'):
    """
    Identifies potential date columns and standardizes their format.
    Returns the dataframe with standardized columns.
    """
    st.subheader("Date Standardization")

    # Identify potential date columns (simplified: look for 'date')
    date_cols = [col for col in df.columns if 'date' in col.lower()]

    standardized_df = df.copy() # Work on a copy
    if date_cols:
        target_date_format = st.text_input("Target date format (e.g., '%Y-%m-%d'):", value='%Y-%m-%d', key="date_format_input")
        for col in standardized_df.columns: # Iterate through all columns in the dataframe copy
             if col in date_cols: # Check if the current column is in the list of date columns
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

    return standardized_df


def perform_standardization():
    """
    Provides UI and logic for performing data standardization on the merged DataFrame
    (from session state) and updates st.session_state.standardized_dataframe.
    """
    st.header("Data Standardization")

    # Access merged dataframe from session state
    merged_df = st.session_state.get('merged_dataframe', None)

    if merged_df is not None and not merged_df.empty:
        st.write("Performing standardization...")

        # 1. Unit Standardization
        df_after_unit_standardization = standardize_units(merged_df)

        # 2. Date Standardization
        df_after_date_standardization = standardize_dates(df_after_unit_standardization)

        standardized_df = df_after_date_standardization

        st.subheader("Standardized Data Preview")
        st.dataframe(standardized_df.head()) # Show preview after standardization steps

        # Store the standardized dataframe in session state
        st.session_state.standardized_dataframe = standardized_df

        # Note: Download button for final result is in app.py, handled in Step 5.

    elif st.session_state.get('loaded_dataframes', {}): # If data is loaded but not merged
        st.info("Please go back to Step 3 to perform Data Merging first.")
    else: # If no data loaded
        st.info("Please go back to Step 1 to upload data.")

    # This function updates session state directly and returns None.
    return None
