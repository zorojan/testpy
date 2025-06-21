import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
import spacy
import re
from pint import UnitRegistry
import os
import google.generativeai as genai
import time
from google.colab import userdata
import numpy as np # Import numpy for combined score calculation

# Get the API key from Colab secrets and configure the API
try:
    google_api_key = userdata.get('GOOGLE_API_KEY')
    if not google_api_key:
        st.error("Google API Key not found in Colab secrets. Please add it.")
        st.stop()
    genai.configure(api_key=google_api_key)
except Exception as e:
    st.error(f"Error configuring Google API: {e}")
    st.stop()


# Define the prompt for the AI model for header comparison
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
    str1 = str(text1).lower().strip() if text1 is not None else ""
    str2 = str(text2).lower().strip() if text2 is not None else ""
    if not str1 or not str2:
        return 0
    return fuzz.ratio(str1, str2)

# Semantic Matching Function
def semantic_match_strings(text1, text2):
    """Performs semantic matching between two strings using SpaCy, returns raw score."""
    str1 = str(text1) if text1 is not None else ""
    str2 = str(text2) if text2 is not None else ""
    if not str1 or not str2:
         return 0.0
    doc1 = st.session_state.nlp(str1)
    doc2 = st.session_state.nlp(str2)
    if doc1.has_vector and doc2.has_vector and len(doc1) > 0 and len(doc2) > 0:
         if doc1.vector.shape == doc2.vector.shape:
              return doc1.similarity(doc2)
         else:
              return 0.0
    return 0.0

# AI Matching Function using Google AI
def ai_match_strings(text1, text2):
    """
    Uses Google AI API to compare two headers and return a similarity score (0-100).
    Handles potential API errors and invalid responses.
    """
    str1 = str(text1).strip() if text1 is not None else ""
    str2 = str(text2).strip() if text2 is not None else ""
    if not str1 or not str2:
        return 0

    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = header_comparison_prompt.format(header1=str1, header2=str2)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                score_text = response.text.strip()
                score = int(score_text)
                score = max(0, min(100, score))
                return score
            except ValueError:
                print(f"Attempt {attempt + 1} failed: API response was not a valid integer: '{response.text}'")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print("Max retries reached. Could not extract valid score.")
                    return -1
            except Exception as e:
                 print(f"Attempt {attempt + 1} failed with API error: {e}")
                 if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                 else:
                    print("Max retries reached. API call failed.")
                    return -1
        return -1
    except Exception as e:
        print(f"An error occurred during the AI matching process: {e}")
        return -1


def perform_header_matching_and_grouping():
    """
    Collects headers from all loaded dataframes (from session state),
    compares all pairs using multiple strategies, groups similar headers based on combined score,
    and displays these groups with interactive selection, showing individual scores.
    Allows user to select representative headers from each source file for each group.
    Stores user-confirmed column mappings in st.session_state.confirmed_column_mappings.
    """
    st.header("Column Header Matching and Grouping")

    loaded_dataframes = st.session_state.get('loaded_dataframes', {})

    if not loaded_dataframes:
        st.info("Please go back to Step 1 to upload data.")
        st.session_state.grouped_header_matches = []
        st.session_state.confirmed_column_mappings = {}
        st.session_state.header_comparisons = [] # Clear comparisons too
        return

    st.subheader("Matching Strategy for Header Grouping")
    # Keep the radio button for consistency, but grouping will use combined score
    matching_strategy_display = st.radio(
        "Choose a primary matching strategy (for display purposes):",
        ('Fuzzy Matching', 'Semantic Matching', 'AI Matching'),
        key="header_grouping_strategy_display"
    )


    # Define threshold input for combined score
    combined_threshold = st.slider("Combined Match Threshold for Grouping (higher is stricter):", 0, 100, 75, key="grouping_combined_threshold") # Using 75 as a starting point


    hide_perfect_matches = st.checkbox("Hide 100% matches", value=True, key="hide_perfect_matches")

    # Optional: Button to trigger comparison calculation only (useful if grouping is slow)
    # if st.button("Calculate All Header Comparisons", key="calculate_comparisons_button"):
    #      st.write("Calculating all-pairs header comparisons...")
    #      # ... (comparison calculation logic) ...
    #      st.session_state.header_comparisons = header_comparisons # Store comparison results
    #      st.write("Comparison calculation completed. Now click 'Group Similar Headers'.")


    if st.button("Group Similar Headers (Based on Combined Score)", key="group_headers_button_combined"):
        st.write(f"Grouping similar headers using Combined Score with threshold {combined_threshold}...")

        all_headers_with_source = []
        unique_headers = set()
        for df_name, df in loaded_dataframes.items():
            for col in df.columns:
                all_headers_with_source.append({'header': col, 'source': df_name})
                unique_headers.add(col)

        if not all_headers_with_source:
            st.write("No headers found in the loaded dataframes.")
            st.session_state.grouped_header_matches = []
            st.session_state.confirmed_column_mappings = {}
            st.session_state.header_comparisons = []
            return

        unique_headers_list = list(unique_headers)
        header_comparisons = []

        # Perform all-pairs comparison with all methods
        st.write("Performing all-pairs header comparisons...")
        comparison_progress = st.progress(0)
        total_comparisons = len(unique_headers_list) * (len(unique_headers_list) - 1) // 2
        completed_comparisons = 0

        for i in range(len(unique_headers_list)):
            for j in range(i + 1, len(unique_headers_list)):
                header1 = unique_headers_list[i]
                header2 = unique_headers_list[j]

                fuzzy_score = fuzzy_match_strings_rapidfuzz(header1, header2)
                semantic_score_raw = semantic_match_strings(header1, header2)
                # Convert semantic score (0.0-1.0) to 0-100 range for combining
                semantic_score = int(semantic_score_raw * 100) if semantic_score_raw is not None else 0

                ai_score = ai_match_strings(header1, header2)

                # Simple combination: average of valid scores (0-100 range)
                scores_for_avg = [s for s in [fuzzy_score, semantic_score, ai_score] if s != -1 and s is not None]
                combined_score = np.mean(scores_for_avg) if scores_for_avg else 0 # Use numpy mean for robustness

                header_comparisons.append({
                    'header1': header1,
                    'header2': header2,
                    'fuzzy_score': fuzzy_score,
                    'semantic_score_raw': semantic_score_raw, # Keep raw semantic score
                    'semantic_score_100': semantic_score, # Store semantic score in 0-100
                    'ai_score': ai_score,
                    'combined_score': combined_score
                })

                completed_comparisons += 1
                if total_comparisons > 0:
                    comparison_progress.progress(completed_comparisons / total_comparisons)


        st.write("Header comparisons completed.")

        # --- Grouping Logic (based on the Combined Score) ---
        grouped_indices = set()
        header_groups = []

        # Iterate through all unique headers to form groups based on combined score
        for i in range(len(unique_headers_list)):
            current_header = unique_headers_list[i]
            if current_header not in grouped_indices:
                current_group_headers = [current_header] # Start a new group
                grouped_indices.add(current_header)

                # Find other headers that match the current header based on the combined score and threshold
                for comparison in header_comparisons:
                    other_header = None
                    if comparison['header1'] == current_header:
                         other_header = comparison['header2']
                    elif comparison['header2'] == current_header:
                         other_header = comparison['header1']

                    if other_header and other_header not in grouped_indices:
                        # Find the combined score for this pair
                        comp_score = None
                        if (comparison['header1'] == current_header and comparison['header2'] == other_header) or \
                           (comparison['header2'] == current_header and comparison['header1'] == other_header):
                            comp_score = comparison.get('combined_score')


                        if comp_score is not None and comp_score != -1: # Check for valid combined score
                             score_passes_threshold = comp_score >= combined_threshold

                             # Check for perfect match based on combined score (assuming 100 is perfect)
                             is_perfect_match = comp_score == 100

                             if score_passes_threshold:
                                 if hide_perfect_matches and is_perfect_match:
                                      pass # Skip adding if hiding perfect matches and it's a perfect match
                                 else:
                                     current_group_headers.append(other_header)
                                     grouped_indices.add(other_header)

                # Convert the group of headers back to the original format including source
                full_group_info = [item for item in all_headers_with_source if item['header'] in current_group_headers]

                # Add the completed group to the list of header groups only if it has more than one item
                # or if hide_perfect_matches is False (to show single items if needed)
                if len(full_group_info) > 1 or (len(full_group_info) == 1 and not hide_perfect_matches):
                     header_groups.append(full_group_info)
                elif len(full_group_info) == 1 and hide_perfect_matches:
                     pass # Skip groups of size 1 when hiding perfect matches


        # Store grouped headers in session state
        st.session_state.grouped_header_matches = header_groups
        st.session_state.last_grouping_strategy = 'Combined Score' # Indicate grouping by combined score
        st.session_state.last_grouping_threshold = combined_threshold
        st.session_state.header_comparisons = header_comparisons # Store comparison results

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
                            # Find the header from this source file with the best combined score against the group name
                            best_header = None
                            best_score = -1
                            representative_header_for_scoring = group_name # Use the first header as representative for scoring

                            for header in headers_from_source_in_group:
                                 # Find the comparison result for this pair
                                 comparison_result = next((comp for comp in header_comparisons
                                                           if (comp['header1'] == representative_header_for_scoring and comp['header2'] == header) or
                                                              (comp['header2'] == representative_header_for_scoring and comp['header1'] == header)), None)

                                 if comparison_result and comparison_result['combined_score'] is not None and comparison_result['combined_score'] != -1: # Use combined score for default selection
                                      score = comparison_result['combined_score']
                                      if score > best_score:
                                           best_score = score
                                           best_header = header
                            # If a best header was found, set it as the default
                            if best_header:
                                 initial_confirmed_mappings[group_name][file_name] = best_header

        st.session_state.confirmed_column_mappings = initial_confirmed_mappings


    # --- Display Grouped Headers and Allow User Selection ---

    if 'grouped_header_matches' in st.session_state and st.session_state.grouped_header_matches:
        st.subheader("Review and Select Headers for Merging")
        st.write("For each suggested group, select the representative header from each source file.")
        st.write("The 'Completed' column indicates if you have selected a header for every source file within that group.")

        grouped_headers = st.session_state.grouped_header_matches
        confirmed_mappings = st.session_state.get('confirmed_column_mappings', {})
        loaded_dataframes = st.session_state.get('loaded_dataframes', {})
        source_files = list(loaded_dataframes.keys())

        # Show detailed comparison scores checkbox outside the loop
        show_detailed_scores = st.checkbox("Show detailed comparison scores", key="show_detailed_scores_checkbox")

        # Display each group using expanders for better organization
        for group_index, group in enumerate(grouped_headers):
            if group:
                group_display_name = group[0]['header'] # Use the first header as representative

                with st.expander(f"Group {group_index + 1}: {group_display_name}"):
                    st.write("Headers in this group:")

                    # Display headers from each source file within the group
                    for file_name in source_files:
                        headers_from_this_source = [item['header'] for item in group if item['source'] == file_name]
                        if headers_from_this_source:
                            st.write(f"- **{file_name}**: {', '.join(headers_from_this_source)}")

                    # Display individual comparison scores if requested
                    if show_detailed_scores and 'header_comparisons' in st.session_state:
                         st.write("Comparison Scores within this group (compared to representative header):")
                         representative_header = group_display_name
                         scores_data = []
                         for item in group:
                             if item['header'] != representative_header:
                                  comparison = next((comp for comp in st.session_state.header_comparisons
                                                     if (comp['header1'] == representative_header and comp['header2'] == item['header']) or
                                                        (comp['header2'] == representative_header and comp['header1'] == item['header'])), None)
                                  if comparison:
                                       scores_data.append({
                                           'Source File': item['source'],
                                           'Header': item['header'],
                                           'Fuzzy Score': comparison.get('fuzzy_score', 'N/A'),
                                           'Semantic Score (0-1)': comparison.get('semantic_score_raw', 'N/A'),
                                           'AI Score': comparison.get('ai_score', 'N/A'),
                                           'Combined Score': f"{comparison.get('combined_score', 'N/A'):.2f}" if comparison.get('combined_score') is not None else 'N/A'
                                       })
                         if scores_data:
                             scores_df = pd.DataFrame(scores_data)
                             st.dataframe(scores_df, use_container_width=True)
                         else:
                              st.info("No comparison scores available for headers within this group against the representative header.")


                    st.write("Select headers for merging:")
                    updated_confirmed_mappings = confirmed_mappings.copy() # Use a copy for updates

                    # Create columns for selectboxes for each source file
                    select_cols = st.columns(len(source_files))
                    group_complete = True # Assume complete until proven otherwise

                    for i, file_name in enumerate(source_files):
                        headers_from_this_source = [item['header'] for item in group if item['source'] == file_name]
                        options = ["-- Select Header --"] + headers_from_this_source
                        current_selection = updated_confirmed_mappings.get(group_display_name, {}).get(file_name, "-- Select Header --")
                        if current_selection not in options:
                             current_selection = "-- Select Header --"

                        selected_header = select_cols[i].selectbox(
                            f"{file_name}", # Label with file name
                            options=options,
                            index=options.index(current_selection) if current_selection in options else 0,
                            key=f"select_{group_index}_{file_name}" # Unique key
                        )

                        # Update the confirmed mapping based on selection
                        if group_display_name not in updated_confirmed_mappings:
                             updated_confirmed_mappings[group_display_name] = {}

                        if selected_header != "-- Select Header --":
                             updated_confirmed_mappings[group_display_name][file_name] = selected_header
                        else:
                             # If "None" is selected, remove the entry for this file
                             if file_name in updated_confirmed_mappings[group_display_name]:
                                  del updated_confirmed_mappings[group_display_name][file_name]
                             group_complete = False # Mark group as incomplete

                    # Display completion status within the expander
                    if group_complete and updated_confirmed_mappings.get(group_display_name):
                        st.write("Group mapping completed: ✅")
                    else:
                        st.write("Group mapping completed: ❌")

                    # Update session state with the changes from this group's selections
                    # This is done within the expander loop, which might be less efficient
                    # but ensures state is updated as selectboxes change.
                    st.session_state.confirmed_column_mappings = updated_confirmed_mappings

        st.markdown("---")
        st.subheader("Confirmed Column Mappings Summary")
        # Display the current confirmed mappings for review
        if st.session_state.confirmed_column_mappings:
            st.json(st.session_state.confirmed_column_mappings) # Display as JSON for clarity
            st.info("These are the mappings based on your selections above. They will be used in the next step.")
        else:
            st.info("No column mappings confirmed yet. Make selections above to confirm.")

        # The detailed comparison scores table is now inside the expander sections per group
        # The overall "Show detailed comparison scores" checkbox controls visibility inside expanders
