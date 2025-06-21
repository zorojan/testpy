import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
import spacy
import re
from pint import UnitRegistry

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

# Fuzzy Matching Functions with improved scoring
def fuzzy_match_strings_improved(text1, text2):
    """Performs multiple fuzzy matching comparisons and returns the max score."""
    # Add checks for None or empty strings at the beginning
    str1 = str(text1).lower().strip() if text1 is not None else ""
    str2 = str(text2).lower().strip() if text2 is not None else ""

    if not str1 or not str2:
        return 0

    score_ratio = fuzz.ratio(str1, str2)
    score_partial_ratio = fuzz.partial_ratio(str1, str2)
    score_token_sort = fuzz.token_sort_ratio(str1, str2)
    score_token_set = fuzz.token_set_ratio(str1, str2)

    # Return the maximum of relevant scores
    return max(score_ratio, score_partial_ratio, score_token_sort, score_token_set)

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
        ('Fuzzy Matching', 'Semantic Matching'),
        key="header_grouping_strategy"
    )

    # Define threshold inputs based on strategy
    threshold = 80 # Default for Fuzzy
    if matching_strategy == 'Fuzzy Matching':
        threshold = st.slider("Match Threshold for Grouping (higher is stricter):", 0, 100, 80, key="grouping_fuzzy_threshold")
    elif matching_strategy == 'Semantic Matching':
        threshold = st.slider("Match Threshold for Grouping (higher is stricter):", 0.0, 1.0, 0.7, key="grouping_semantic_threshold", step=0.05)

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
            match_func = lambda text1, text2: fuzzy_match_strings_improved(text1, text2)
            score_attribute = 'fuzzy_score'
        elif matching_strategy == 'Semantic Matching':
             if 'nlp' in st.session_state and st.session_state.nlp:
                 match_func = lambda text1, text2: semantic_match_strings(text1, text2) # Get raw score
                 score_attribute = 'semantic_score'
             else:
                 st.warning("SpaCy model not loaded. Cannot perform Semantic matching.")
                 match_func = None


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
                            if score is not None:
                                 score_passes_threshold = score >= threshold
                                 is_perfect_match = False
                                 if matching_strategy == 'Fuzzy Matching' and score == 100:
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
                         # it remains a group of 1. We skip adding it if it's a perfect self-match.
                         pass # Skip groups of size 1 when hiding perfect matches


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
                                     if score is not None and score > best_score:
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
            if not group: continue

            # Determine a representative name for the group (e.g., the first header in the group)
            group_name = group[0]['header']

            # Create columns for this group's row
            group_cols = st.columns(num_cols)

            # Display group name
            group_cols[0].write(f"**{group_name}**")

            # For each source file, create a selectbox
            current_group_mappings = {} # Store mappings for this specific group
            source_files_in_group = list(set([item['source'] for item in group])) # Get unique source files in this group
            for i, file_name in enumerate(source_files): # Iterate through ALL source files for consistent column layout
                 # Only display selectbox if this source file has headers in this group
                 if file_name in source_files_in_group:
                      headers_from_source = [item['header'] for item in group if item['source'] == file_name]

                      # Add an option for "Do not merge" or similar if no header from this source is in the group
                      options = ["- Select Header -"] + headers_from_source

                      # Determine the default selection based on confirmed_mappings or initial best match
                      default_index = 0 # Default to the placeholder
                      # Check if a mapping is already confirmed for this group and file
                      if group_name in confirmed_mappings and file_name in confirmed_mappings[group_name]:
                           selected_header_value = confirmed_mappings[group_name][file_name]
                           if selected_header_value in options:
                                default_index = options.index(selected_header_value)
                           # else: default remains 0 (placeholder) if the previously selected header is somehow not in current options
                      else:
                           # If no confirmed mapping, use the initial best match determined after grouping
                           # Find the best header from this file in this group based on initial scoring
                           best_header = None
                           best_score = -1
                           representative_header_for_scoring = group_name # Use the first header as representative for scoring
                           # Re-run the scoring logic to find the best header from this specific source file in this group
                           if match_func: # Ensure match_func is available
                                for header in headers_from_source:
                                     score = match_func(representative_header_for_scoring, header)
                                     if score is not None and score > best_score:
                                          best_score = score
                                          best_header = header
                                if best_header and best_header in options:
                                      default_index = options.index(best_header)


                      # Use a unique key for each selectbox
                      selected_header = group_cols[i+1].selectbox(
                          "Select header:",
                          options,
                          index=default_index, # Set default selection
                          key=f"group_{group_index}_file_{file_name}_select"
                      )

                      # Store the selected header if it's not the placeholder
                      if selected_header not in ["- Select Header -", "- No matching header in this file -"]:
                           current_group_mappings[file_name] = selected_header
                 else:
                     # If the source file is not in this group, display an empty cell or placeholder
                     group_cols[i+1].write("") # Or st.empty()


            # Update the confirmed mappings for this group
            if group_name not in updated_confirmed_mappings:
                 updated_confirmed_mappings[group_name] = {}
            updated_confirmed_mappings[group_name].update(current_group_mappings)

            # Display completion status
            # Check if this specific group's mapping is complete based on updated_confirmed_mappings
            is_group_complete_updated = True
            # Check if a selection has been made for ALL source files that have headers in this group
            for file_name in source_files_in_group:
                 if group_name not in updated_confirmed_mappings or file_name not in updated_confirmed_mappings[group_name]:
                      is_group_complete_updated = False
                      break
            group_cols[num_cols - 1].checkbox("Done", value=is_group_complete_updated, disabled=True, key=f"group_{group_index}_complete_checkbox")


        # After iterating through all groups, update the session state with the new selections
        st.session_state.confirmed_column_mappings = updated_confirmed_mappings

        st.markdown("---") # Separator

        st.subheader("Confirmed Column Mappings Summary")
        if confirmed_mappings: # Display the confirmed mappings based on the session state
             # Display the confirmed mappings
             confirmed_df = pd.DataFrame({
                 'Consolidated Column': list(confirmed_mappings.keys()),
                 'Source Mappings': [str(m) for m in confirmed_mappings.values()] # Display mappings as string for simplicity
             })
             st.dataframe(confirmed_df)
             st.info("These mappings will be used in the Data Merging step.")
        else:
             st.info("No headers have been selected for merging yet.")


    # This function updates session state directly and returns None.
    return None
