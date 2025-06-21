import streamlit as st
import pandas as pd
from rapidfuzz import fuzz # Import rapidfuzz
import spacy
import re
from pint import UnitRegistry
import os # Import os to write to file
import google.generativeai as genai # Import genai
import time # Import time for retries

# Define the prompt for the AI model for header comparison (assuming it's already defined)
# This prompt needs to be defined or imported here if not globally available
header_comparison_prompt = """
Compare the two provided column headers and assess their similarity for the purpose of merging data.
Your task is to provide a similarity score as an integer between 0 and 100, where 100 means the headers are identical or highly similar in meaning and refer to the same type of data, and 0 means they are completely unrelated.

Consider variations in:
- Spelling (e.g., "colour" vs "color")
- Case ("Product Name" vs "product name")
- Spacing and special characters ("Product-ID" vs "Product ID")
- Common abbreviations ("Prod. No." vs "Product Number")
- Synonyms or closely related terms ("Customer Identifier" vs "Client ID")
- Order of words if the meaning is the same ("Shipping Address" vs "Address, Shipping")

The output should ONLY be the integer score, nothing else.

Examples:

Header 1: "Product Name"
Header 2: "Product Name"
Score: 100

Header 1: "Customer ID"
Header 2: "Client Identifier"
Score: 95

Header 1: "Shipping Address"
Header 2: "Delivery Location"
Score: 85

Header 1: "Order Date"
Header 2: "Purchase Timestamp"
Score: 70

Header 1: "Product Cost"
Header 2: "Price (USD)"
Score: 60

Header 1: "Country"
Header 2: "Region"
Score: 50

Header 1: "Email"
Header 2: "Phone Number"
Score: 10

Header 1: "Order ID"
Header 2: "Supplier Name"
Score: 0

Header 1: ""
Header 2: "Product Name"
Score: 0

Header 1: "Item Number"
Header 2: "Numéro d'article"
Score: 50 # Partial match due to language difference

Header 1: "QTY"
Header 2: "Quantity Ordered"
Score: 90

Compare the following two headers:
Header 1: {header1}
Header 2: {header2}
Score:
"""

# Load SpaCy model
# Use session state to load the model once
if 'nlp' not in st.session_state:
    try:
        st.session_state.nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model 'en_core_web_sm' not found. Please run '!python -m spacy download en_core_web_sm' in a separate cell.")
        st.stop()

# Initialize Pint UnitRegistry (although not strictly used in header matching, keep for consistency if other matching methods are added later)
ureg = UnitRegistry()

# RapidFuzz Matching Functions
def fuzzy_match_strings_rapidfuzz(text1, text2):
    """Performs fuzzy matching using rapidfuzz and returns the ratio score."""
    # Add checks for None or empty strings at the beginning
    str1 = str(text1).lower().strip() if text1 is not None else ""
    str2 = str(text2).lower().strip() if text2 is not None else ""

    if not str1 or not str2:
        return 0

    # Using the simple ratio from rapidfuzz
    score = fuzz.ratio(str1, str2)

    return score

# Semantic Matching Function
def semantic_match_strings(text1, text2):
    """Performs semantic matching between two strings using SpaCy, returns raw score."""
    # Add checks for None or empty strings
    str1 = str(text1) if text1 is not None else ""
    str2 = str(text2) if text2 is not None else ""

    if not str1 or not str2:
         return 0.0 # Return 0 if strings are empty

    doc1 = st.session_state.nlp(str1)
    doc2 = st.session_state.nlp(str2)

    if doc1.has_vector and doc2.has_vector and len(doc1) > 0 and len(doc2) > 0:
         if doc1.vector.shape == doc2.vector.shape:
              return doc1.similarity(doc2)
         else:
              return 0.0 # Return 0 if dimensions mismatch or no vectors
    return 0.0 # Return 0 if docs are empty or no vectors

# AI Matching Function using Google AI
def ai_match_strings(text1, text2):
    """
    Uses Google AI API to compare two headers and return a similarity score (0-100).
    Handles potential API errors and invalid responses.
    """
    str1 = str(text1).strip() if text1 is not None else ""
    str2 = str(text2).strip() if text2 is not None else ""

    if not str1 or not str2:
        return 0 # Return 0 for empty strings

    try:
        # Instantiate the model
        # Use a robust model like gemini-pro
        model = genai.GenerativeModel('gemini-pro')

        # Format the prompt with the input headers
        prompt = header_comparison_prompt.format(header1=str1, header2=str2)

        # Call the API
        # Add a retry mechanism for potential transient errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                # Process the response to extract the integer score
                # Expecting only an integer in the response
                score_text = response.text.strip()
                score = int(score_text)

                # Ensure the score is within the 0-100 range
                score = max(0, min(100, score))
                return score

            except ValueError:
                # Handle cases where the response is not a valid integer
                print(f"Attempt {attempt + 1} failed: API response was not a valid integer: '{response.text}'")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                    continue
                else:
                    print("Max retries reached. Could not extract valid score.")
                    return -1 # Indicate failure

            except Exception as e:
                 # Handle other potential API errors
                 print(f"Attempt {attempt + 1} failed with API error: {e}")
                 if attempt < max_retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                    continue
                 else:
                    print("Max retries reached. API call failed.")
                    return -1 # Indicate failure

        return -1 # Return -1 if all retries fail

    except Exception as e:
        print(f"An error occurred during the AI matching process: {e}")
        return -1 # Indicate failure outside of API call


def perform_header_matching_and_grouping():
    """
    Collects headers from all loaded dataframes (from session state),
    groups similar headers, and displays these groups with interactive selection.
    Allows user to select representative headers from each source file for each group.
    Stores user-confirmed column mappings in st.session_state.confirmed_column_mappings.
    """
    st.header("Column Header Matching and Grouping")

    # Access loaded dataframes directly from session state
    loaded_dataframes = st.session_state.get('loaded_dataframes', {})

    if not loaded_dataframes:
        st.info("Please go back to Step 1 to upload data.")
        # Clear any previous grouping results if no data loaded
        st.session_state.grouped_header_matches = []
        st.session_state.confirmed_column_mappings = {} # Also clear confirmed mappings
        return # Exit the function if no dataframes are loaded

    st.subheader("Matching Strategy for Header Grouping")
    matching_strategy = st.radio(
        "Choose a matching strategy for grouping headers:",
        ('Fuzzy Matching', 'Semantic Matching', 'AI Matching'), # Added AI Matching option
        key="header_grouping_strategy"
    )

    # Define threshold inputs based on strategy
    threshold = 80 # Default for Fuzzy and AI
    if matching_strategy == 'Fuzzy Matching':
        threshold = st.slider("Match Threshold for Grouping (higher is stricter):", 0, 100, 80, key="grouping_fuzzy_threshold")
    elif matching_strategy == 'Semantic Matching':
        threshold = st.slider("Match Threshold for Grouping (higher is stricter):", 0.0, 1.0, 0.7, key="grouping_semantic_threshold", step=0.05)
    elif matching_strategy == 'AI Matching': # Added threshold for AI Matching
        threshold = st.slider("Match Threshold for Grouping (higher is stricter):", 0, 100, 80, key="grouping_ai_threshold")


    # Option to hide 100% matches
    hide_perfect_matches = st.checkbox("Hide 100% matches", value=True, key="hide_perfect_matches")

    # Button to trigger grouping
    if st.button("Group Similar Headers", key="group_headers_button"):
        st.write(f"Grouping similar headers using {matching_strategy} with threshold {threshold}...")

        # Collect all headers with their source dataframe names
        all_headers_with_source = []
        for df_name, df in loaded_dataframes.items():
            for col in df.columns:
                all_headers_with_source.append({'header': col, 'source': df_name})

        if not all_headers_with_source:
            st.write("No headers found in the loaded dataframes.")
            st.session_state.grouped_header_matches = [] # Store empty list
            st.session_state.confirmed_column_mappings = {}
            return

        # Use a list to keep track of which headers have already been grouped
        grouped_indices = set()
        header_groups = [] # Reset header_groups

        # Determine which matching function to use
        match_func = None
        score_attribute = None # Attribute name for score in the matching results
        if matching_strategy == 'Fuzzy Matching':
            match_func = lambda text1, text2: fuzzy_match_strings_rapidfuzz(text1, text2) # Use rapidfuzz function
            score_attribute = 'fuzzy_score'
        elif matching_strategy == 'Semantic Matching':
             if 'nlp' in st.session_state and st.session_state.nlp:
                 match_func = lambda text1, text2: semantic_match_strings(text1, text2) # Get raw score
                 score_attribute = 'semantic_score'
             else:
                 st.warning("SpaCy model not loaded. Cannot perform Semantic matching.")
                 match_func = None
        elif matching_strategy == 'AI Matching': # Assigned AI matching function
             match_func = lambda text1, text2: ai_match_strings(text1, text2)
             score_attribute = 'ai_score'


        if match_func:
            # Iterate through all headers to form groups
            for i in range(len(all_headers_with_source)):
                if i not in grouped_indices:
                    current_header_info = all_headers_with_source[i]
                    current_group = [current_header_info] # Start a new group with this header
                    grouped_indices.add(i) # Mark this header as grouped

                    # Compare this header with all subsequent headers that haven't been grouped yet
                    for j in range(i + 1, len(all_headers_with_source)):
                        if j not in grouped_indices:
                            compare_header_info = all_headers_with_source[j]
                            score = match_func(current_header_info['header'], compare_header_info['header'])

                            # Apply the threshold for grouping, considering hide_perfect_matches
                            if score is not None and score != -1: # Check for valid score (not -1 indicating AI failure)
                                 score_passes_threshold = score >= threshold
                                 is_perfect_match = False
                                 if (matching_strategy == 'Fuzzy Matching' or matching_strategy == 'AI Matching') and score == 100:
                                      is_perfect_match = True
                                 elif matching_strategy == 'Semantic Matching' and score == 1.0:
                                      is_perfect_match = True


                                 if score_passes_threshold:
                                      if hide_perfect_matches and is_perfect_match:
                                           # If hiding perfect matches, and it's a perfect match, skip adding to this group
                                           pass # Don't add to current group, it will form its own group later if not already grouped
                                      else:
                                          current_group.append(compare_header_info) # Add to the current group
                                          grouped_indices.add(j) # Mark as grouped

                    # Add the completed group to the list of header groups only if it has more than one item
                    # or if hide_perfect_matches is False (to show single items if needed)
                    if len(current_group) > 1 or not hide_perfect_matches:
                         header_groups.append(current_group)
                    elif len(current_group) == 1 and hide_perfect_matches:
                         # If hide_perfect_matches is True, and a header didn't find a non-perfect match,
                         # it remains a group of 1. We skip groups of size 1 when hiding perfect matches
                         pass


            # Store grouped headers in session state
            st.session_state.grouped_header_matches = header_groups
            st.session_state.last_grouping_strategy = matching_strategy
            st.session_state.last_grouping_threshold = threshold
            # Initialize confirmed mappings based on suggested groups with default selections
            initial_confirmed_mappings = {}
            for group in header_groups:
                 if group:
                      # Determine a representative name for the group (e.g., the first header in the group)
                      group_name = group[0]['header']
                      initial_confirmed_mappings[group_name] = {}
                      # For each source file in this group, find the best matching header
                      source_files_in_group = list(set([item['source'] for item in group]))
                      for file_name in source_files_in_group:
                           headers_from_source_in_group = [item['header'] for item in group if item['source'] == file_name]
                           if headers_from_source_in_group:
                                # Find the header with the best match score against the group name (or a representative header)
                                best_header = None
                                best_score = -1
                                representative_header_for_scoring = group_name # Use the first header as representative for scoring
                                for header in headers_from_source_in_group:
                                     score = match_func(representative_header_for_scoring, header)
                                     if score is not None and score != -1:
                                          if score > best_score:
                                               best_score = score
                                               best_header = header
                                # If a best header was found, set it as the default
                                if best_header:
                                     initial_confirmed_mappings[group_name][file_name] = best_header

            st.session_state.confirmed_column_mappings = initial_confirmed_mappings


        else:
             st.warning("Matching function could not be initialized.")

    # --- Display Grouped Headers and Allow User Selection ---

    # Check if grouping results are available in session state
    if 'grouped_header_matches' in st.session_state and st.session_state.grouped_header_matches:
        st.subheader("Review and Select Headers for Merging")
        st.write("For each suggested group, select the representative header from each source file.")

        grouped_headers = st.session_state.grouped_header_matches
        confirmed_mappings = st.session_state.get('confirmed_column_mappings', {}) # Get current selections

        # Create columns for header name and each source file
        source_files = list(loaded_dataframes.keys())
        # Determine the number of columns needed: 1 for Group Name + 1 for each Source File + 1 for Completion Status
        num_cols = 1 + len(source_files) + 1
        cols = st.columns(num_cols)

        # Display header row
        cols[0].write("**Group Name**")
        for i, file_name in enumerate(source_files):
            cols[i+1].write(f"**{file_name}**")
        cols[num_cols - 1].write("**Completed**") # Header for completion status

        st.markdown("---") # Separator

        # Display each group with selectboxes
        updated_confirmed_mappings = {} # Temporary dict to store selections from UI
        for group_index, group in enumerate(grouped_headers):
            if group:
                # Use the first header in the group as the default group name for display
                group_display_name = group[0]['header']
                # Ensure the group name exists in confirmed_mappings if it's the first run
                if group_display_name not in confirmed_mappings:
                    confirmed_mappings[group_display_name] = {}

                group_cols = st.columns(num_cols)
                group_cols[0].write(f"**{group_display_name}**") # Display the group name

                all_headers_in_group = [item['header'] for item in group]

                # For each source file, create a selectbox with headers from that file within this group
                group_complete = True # Assume group is complete until proven otherwise
                for i, file_name in enumerate(source_files):
                    headers_from_this_source = [item['header'] for item in group if item['source'] == file_name]
                    # Add an option for "None" if a file doesn't have a header in this group or if the user wants to exclude it
                    options = ["-- Select Header --"] + headers_from_this_source
                    # Determine the default selection
                    current_selection = confirmed_mappings.get(group_display_name, {}).get(file_name, "-- Select Header --")
                    if current_selection not in options:
                         current_selection = "-- Select Header --" # Reset if the previously selected header is no longer an option

                    selected_header = group_cols[i+1].selectbox(
                        f"Select for {file_name}",
                        options=options,
                        index=options.index(current_selection) if current_selection in options else 0,
                        key=f"select_{group_index}_{file_name}" # Unique key for each selectbox
                    )

                    # Store the selected mapping
                    if group_display_name not in updated_confirmed_mappings:
                         updated_confirmed_mappings[group_display_name] = {}

                    if selected_header != "-- Select Header --":
                         updated_confirmed_mappings[group_display_name][file_name] = selected_header
                    else:
                         # If "None" is selected, ensure the entry for this file is removed from confirmed mappings for this group
                         if file_name in updated_confirmed_mappings[group_display_name]:
                              del updated_confirmed_mappings[group_display_name][file_name]
                         group_complete = False # Mark group as incomplete if any file is not mapped

                # Display completion status
                if group_complete and updated_confirmed_mappings.get(group_display_name):
                    group_cols[num_cols - 1].write("✅")
                else:
                     group_cols[num_cols - 1].write("❌")


        # Update confirmed mappings in session state based on user selections
        st.session_state.confirmed_column_mappings = updated_confirmed_mappings

        st.markdown("---") # Separator
        st.subheader("Confirmed Column Mappings")
        # Display the current confirmed mappings for review
        if st.session_state.confirmed_column_mappings:
            st.json(st.session_state.confirmed_column_mappings) # Display as JSON for clarity
            st.info("These are the mappings based on your selections above. They will be used in the next step.")
        else:
            st.info("No column mappings confirmed yet. Make selections above to confirm.")

# Note: You will need to rerun the Streamlit app cell (cell bN5HEtVVTcOy) after
# updating matching.py for the changes to take effect in the running app.
