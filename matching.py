import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
import spacy
import re
from pint import UnitRegistry
import os
import google.generativeai as genai
import time
import numpy as np

# Initialize session state variables at the beginning
if 'nlp' not in st.session_state:
    try:
        st.session_state.nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model 'en_core_web_sm' not found. Please run '!python -m spacy download en_core_web_sm' in a separate cell.")
        st.stop()

if 'loaded_dataframes' not in st.session_state:
     st.session_state.loaded_dataframes = {}

if 'grouped_header_matches' not in st.session_state:
    st.session_state.grouped_header_matches = []

if 'confirmed_column_mappings' not in st.session_state:
    st.session_state.confirmed_column_mappings = {}

if 'header_comparisons' not in st.session_state:
    st.session_state.header_comparisons = []


# Get the API key from environment variables and configure the API
try:
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        st.error("Google API Key not found in environment variables. AI matching will be disabled.")
        print("Warning: GOOGLE_API_KEY not found in environment variables. AI matching will return -1.")
        genai_configured = False
    else:
        genai.configure(api_key=google_api_key)
        genai_configured = True

except Exception as e:
    st.error(f"Error configuring Google API: {e}")
    genai_configured = False


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
         # print(f"Debugging Semantic: Empty string for '{text1}' or '{text2}', returning 0.0")
         return 0.0
    if 'nlp' in st.session_state and st.session_state.nlp:
        doc1 = st.session_state.nlp(str1)
        doc2 = st.session_state.nlp(str2)

        # Add debugging for vector presence and length
        # print(f"Debugging Semantic: Header 1 '{str1}' - has_vector: {doc1.has_vector}, len: {len(doc1)}")
        # print(f"Debugging Semantic: Header 2 '{str2}' - has_vector: {doc2.has_vector}, len: {len(doc2)}")


        if doc1.has_vector and doc2.has_vector and len(doc1) > 0 and len(doc2) > 0:
             if doc1.vector.shape == doc2.vector.shape:
                  score = doc1.similarity(doc2)
                  # print(f"Debugging Semantic: Similarity score for '{str1}' and '{str2}': {score}")
                  return score
             else:
                  # print(f"Debugging Semantic: Vector shape mismatch for '{str1}' and '{str2}', returning 0.0")
                  return 0.0
        else:
             # print(f"Debugging Semantic: Missing vector or zero length doc for '{str1}' or '{str2}', returning 0.0")
             return 0.0
    else:
        print("Debugging Semantic: SpaCy model not loaded in session state.")
        return 0.0 # Should not happen with initialization at top

# AI Matching Function using Google AI
def ai_match_strings(text1, text2):
    """
    Uses Google AI API to compare two headers and return a similarity score (0-100).
    Handles potential API errors and invalid responses.
    """
    if not globals().get('genai_configured', False):
         print("Debugging AI: AI Matching skipped: Google AI API not configured.")
         return -1

    str1 = str(text1).strip() if text1 is not None else ""
    str2 = str(text2).strip() if text2 is not None else ""
    if not str1 or not str2:
        print(f"Debugging AI: Empty string for '{text1}' or '{str2}', returning 0")
        return 0

    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = header_comparison_prompt.format(header1=str1, header2=str2)
        # print(f"Debugging AI: Sending prompt for '{str1}' vs '{str2}': {prompt}") # Too verbose

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                score_text = response.text.strip()
                # print(f"Debugging AI: Received raw response for '{str1}' vs '{str2}': '{response.text}'") # Keep for debugging API response format

                # Attempt to parse the score
                score = int(score_text)
                score = max(0, min(100, score))
                # print(f"Debugging AI: Parsed score for '{str1}' vs '{str2}': {score}") # Keep for debugging parsed score
                return score
            except ValueError:
                print(f"Debugging AI: Attempt {attempt + 1} failed: API response was not a valid integer: '{response.text}' for '{str1}' vs '{str2}'")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"Debugging AI: Max retries reached. Could not extract valid score for '{str1}' vs '{str2}'. Returning -1.")
                    return -1
            except Exception as e:
                 print(f"Debugging AI: Attempt {attempt + 1} failed with API error: {e} for '{str1}' vs '{str2}'")
                 if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                 else:
                    print(f"Debugging AI: Max retries reached. API call failed for '{str1}' vs '{str2}'. Returning -1.")
                    return -1
        return -1 # Should not be reached if max_retries > 0
    except Exception as e:
        print(f"Debugging AI: An error occurred during the AI matching process for '{str1}' vs '{str2}': {e}")
        return -1


def perform_header_matching_and_grouping():
    """
    Collects headers from all loaded dataframes (from session state),
    compares all pairs using multiple strategies, groups similar headers based on selected strategy,
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
    # Added back radio buttons for strategy selection
    matching_strategy = st.radio(
        "Choose a matching strategy for grouping headers:",
        ('Fuzzy Matching', 'Semantic Matching', 'AI Matching', 'Combined Score'), # Added Combined option
        key="header_grouping_strategy"
    )

    # Added back threshold slider based on the selected strategy
    threshold = 80 # Default for Fuzzy and AI
    if matching_strategy == 'Fuzzy Matching':
        threshold = st.slider("Match Threshold for Grouping (higher is stricter):", 0, 100, 80, key="grouping_fuzzy_threshold")
    elif matching_strategy == 'Semantic Matching':
        # Semantic threshold is typically 0.0 to 1.0
        # Need a separate key for semantic threshold slider to avoid conflicts
        threshold = st.slider("Match Threshold for Grouping (higher is stricter):", 0.0, 1.0, 0.7, key="grouping_semantic_threshold", step=0.05)
    elif matching_strategy == 'AI Matching':
        threshold = st.slider("Match Threshold for Grouping (higher is stricter):", 0, 100, 80, key="grouping_ai_threshold")
    elif matching_strategy == 'Combined Score': # Threshold for Combined Score
        threshold = st.slider("Combined Match Threshold for Grouping (higher is stricter):", 0, 100, 60, key="grouping_combined_threshold") # Default 60

    hide_perfect_matches = st.checkbox("Hide 100% matches", value=True, key="hide_perfect_matches")


    # --- Moved Comparison and Grouping Logic Outside the Button ---
    # This ensures it runs on every rerun triggered by UI changes
    # The button can now be used to force a rerun if needed, but isn't required for initial grouping/updates

    # Optional: Add a button to manually trigger grouping if automatic is slow
    # if st.button("Recalculate Grouping"):
    #     st.write("Recalculating...")
    #     # The logic below will run anyway, but this button can be a visual cue or force a rerun


    st.write("Performing all-pairs header comparisons and grouping...")

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
    with st.status("Calculating comparisons...", expanded=True) as status:
        comparison_progress = st.progress(0)
        total_comparisons = len(unique_headers_list) * (len(unique_headers_list) - 1) // 2
        completed_comparisons = 0

        for i in range(len(unique_headers_list)):
            for j in range(i + 1, len(unique_headers_list)):
                header1 = unique_headers_list[i]
                header2 = unique_headers_list[j]

                fuzzy_score = fuzzy_match_strings_rapidfuzz(header1, header2)
                semantic_score_raw = semantic_match_strings(header1, header2)
                semantic_score = int(semantic_score_raw * 100) if semantic_score_raw is not None else 0

                ai_score = ai_match_strings(header1, header2)


                # --- Combined Score Calculation ---
                valid_scores = [s for s in [fuzzy_score, semantic_score, ai_score] if s != -1 and s is not None]
                combined_score = max(valid_scores) if valid_scores else 0 # Using Maximum Score


                header_comparisons.append({
                    'header1': header1,
                    'header2': header2,
                    'fuzzy_score': fuzzy_score,
                    'semantic_score_raw': semantic_score_raw,
                    'semantic_score_100': semantic_score,
                    'ai_score': ai_score,
                    'combined_score': combined_score # Use the chosen combined score
                })

                completed_comparisons += 1
                if total_comparisons > 0:
                    comparison_progress.progress(completed_comparisons / total_comparisons)

        status.update(label="Header comparisons completed!", state="complete", expanded=False)


    # --- Grouping Logic (based on the Selected Strategy) ---
    grouped_indices = set()
    header_groups = []

    # Determine which score attribute and threshold to use based on the selected strategy
    score_attribute_for_grouping = None
    threshold_for_grouping = threshold
    if matching_strategy == 'Fuzzy Matching':
        score_attribute_for_grouping = 'fuzzy_score'
    elif matching_strategy == 'Semantic Matching':
        score_attribute_for_grouping = 'semantic_score_raw' # Use raw score for Semantic
    elif matching_strategy == 'AI Matching':
        score_attribute_for_grouping = 'ai_score'
    elif matching_strategy == 'Combined Score':
        score_attribute_for_grouping = 'combined_score'


    if score_attribute_for_grouping:
         st.write(f"Performing grouping using {matching_strategy} with threshold {threshold}...")
         grouping_progress = st.progress(0)
         total_unique_headers = len(unique_headers_list)
         processed_headers_for_grouping = 0


         for i in range(len(unique_headers_list)):
             current_header = unique_headers_list[i]
             if current_header not in grouped_indices:
                 current_group_headers = [current_header]
                 grouped_indices.add(current_header)

                 for comparison in header_comparisons:
                     other_header = None
                     if comparison['header1'] == current_header:
                          other_header = comparison['header2']
                     elif comparison['header2'] == current_header:
                          other_header = comparison['header1']

                     if other_header and other_header not in grouped_indices:
                         # Get the score for the selected strategy
                         comp_score = comparison.get(score_attribute_for_grouping)

                         # Handle Semantic score (float 0.0-1.0) vs others (int 0-100)
                         score_passes_threshold = False
                         if matching_strategy == 'Semantic Matching':
                             if comp_score is not None:
                                 score_passes_threshold = comp_score >= threshold_for_grouping
                                 # Debugging Semantic Grouping Threshold Check
                                 # print(f"Debugging Grouping: Semantic check for '{current_header}' vs '{other_header}': Score {comp_score}, Threshold {threshold_for_grouping}, Pass {score_passes_threshold}")
                         elif comp_score is not None and comp_score != -1: # Check for valid int/combined score
                             score_passes_threshold = comp_score >= threshold_for_grouping
                             # Debugging Int/Combined Grouping Threshold Check
                             # print(f"Debugging Grouping: Int/Combined check for '{current_header}' vs '{other_header}': Score {comp_score}, Threshold {threshold_for_grouping}, Pass {score_passes_threshold}")


                         is_perfect_match = False
                         if matching_strategy == 'Fuzzy Matching' and comp_score == 100:
                              is_perfect_match = True
                         elif matching_strategy == 'Semantic Matching' and comp_score is not None and comp_score >= 1.0: # Check for >= 1.0 for float perfect match
                              is_perfect_match = True
                         elif matching_strategy == 'AI Matching' and comp_score == 100:
                              is_perfect_match = True
                         elif matching_strategy == 'Combined Score' and comp_score == 100: # Assuming 100 is perfect for combined
                              is_perfect_match = True


                         if score_passes_threshold:
                             if is_perfect_match and hide_perfect_matches:
                                  # Debugging Hide Perfect Matches
                                  # print(f"Debugging Grouping: Hiding perfect match '{current_header}' vs '{other_header}'")
                                  pass
                             else:
                                 current_group_headers.append(other_header)
                                 grouped_indices.add(other_header)
                                 # print(f"Debugging Grouping: Adding '{other_header}' to group with '{current_header}'")


         full_group_info = [item for item in all_headers_with_source if item['header'] in current_group_headers]

         # Only add groups with more than one header unless not hiding perfect matches
         # This is to avoid showing every single header as its own group when hide_perfect_matches is True
         if len(full_group_info) > 1 or (len(full_group_info) == 1 and not hide_perfect_matches):
              header_groups.append(full_group_info)
              # print(f"Debugging Grouping: Finalizing group starting with '{current_header}' with {len(full_group_info)} items.")


         processed_headers_for_grouping += 1
         if total_unique_headers > 0:
             grouping_progress.progress(processed_headers_for_grouping / total_unique_headers)
        else:
            st.warning("Invalid matching strategy selected for grouping.")
            header_groups = [] # Clear groups if strategy is invalid


    st.write("Grouping completed.")


    st.session_state.grouped_header_matches = header_groups
    st.session_state.last_grouping_strategy = matching_strategy # Store the selected strategy
    st.session_state.last_grouping_threshold = threshold # Store the selected threshold
    st.session_state.header_comparisons = header_comparisons # Store comparison results

    # Initialize confirmed mappings based on suggested groups with default selections
    # This part should run after grouping is complete
    initial_confirmed_mappings = {}
    for group in header_groups:
         if group:
              # Determine a representative name for the group (e.g., the first header in the group)
              representative_header_for_group = group[0]['header']
              initial_confirmed_mappings[representative_header_for_group] = {}
              source_files_in_group = list(set([item['source'] for item in group]))

              for file_name in source_files_in_group:
                   headers_from_source_in_group = [item['header'] for item in group if item['source'] == file_name]
                   if headers_from_source_in_group:
                        # Find the header from this source file with the best combined score against the representative header
                        best_header = None
                        best_score = -1

                        for header in headers_from_source_in_group:
                             comparison_result = next((comp for comp in header_comparisons
                                                       if (comp['header1'] == representative_header_for_group and comp['header2'] == header) or
                                                          (comp['header2'] == header and comp['header1'] == representative_header_for_group)), None) # Corrected: Ensured comparison is found regardless of order


                             # Use combined score for default selection, regardless of grouping strategy
                             if comparison_result and comparison_result['combined_score'] is not None and comparison_result['combined_score'] != -1:
                                  score = comparison_result['combined_score']
                                  if score > best_score:
                                       best_score = score
                                       best_header = header
                        if best_header:
                            initial_confirmed_mappings[representative_header_for_group][file_name] = best_header
                            # print(f"Debugging Selection: Defaulting '{file_name}' in group '{representative_header_for_group}' to '{best_header}' (Combined Score: {best_score})")


    st.session_state.confirmed_column_mappings = initial_confirmed_mappings


    # --- Display Grouped Headers and Allow User Selection ---

    # Display the results only if grouping has been performed and there are groups
    if 'grouped_header_matches' in st.session_state and st.session_state.grouped_header_matches:
        st.subheader("Review and Select Headers for Merging")
        st.write("For each suggested group, select the representative header from each source file.")
        st.write("The 'Completed' column indicates if you have selected a header for every source file within that group.")

        grouped_headers = st.session_state.grouped_header_matches
        # Use confirmed_column_mappings from session state directly for display and modification
        confirmed_mappings = st.session_state.confirmed_column_mappings
        loaded_dataframes = st.session_state.get('loaded_dataframes', {})
        source_files = list(loaded_dataframes.keys())

        # Use expanders for each group
        for group_index, group in enumerate(grouped_headers):
            if group:
                # Use the representative header chosen during initial mapping for the expander label
                # Fallback to the first header if no representative was mapped (unlikely but safe)
                group_display_name = group[0]['header']
                # Try to find the representative header from confirmed mappings for a more stable label
                # This is tricky because the group name key in confirmed_mappings is based on the first header
                # We can just use the first header as the label, it's consistent with grouping logic
                # representative_header_from_mapping = next((k for k, v in confirmed_mappings.items() if any(item['header'] in v.values() for item in group)), group_display_name)


                group_complete = True

                source_files_in_group = list(set([item['source'] for item in group]))
                # Check completion based on the group name key in confirmed_mappings
                for file_name in source_files_in_group:
                     if file_name not in confirmed_mappings.get(group_display_name, {}) or \
                        confirmed_mappings.get(group_display_name, {}).get(file_name) == "-- Select Header --":
                         group_complete = False
                         break

                expander_label = f"Group {group_index + 1}: {group_display_name}"
                if group_complete:
                    expander_label += " ✅"
                else:
                    expander_label += " ❌"

                with st.expander(expander_label):
                    st.write("Select the header from each source file that corresponds to this group.")

                    # Use columns within expander for source file and selectbox
                    expander_cols = st.columns(len(source_files))

                    # Initialize mapping for this group if not exists (should be done during grouping, but defensive)
                    if group_display_name not in st.session_state.confirmed_column_mappings:
                         st.session_state.confirmed_column_mappings[group_display_name] = {}


                    for i, file_name in enumerate(source_files):
                         headers_from_this_source = [item['header'] for item in group if item['source'] == file_name]
                         options = ["-- Select Header --"] + headers_from_this_source
                         # Get current selection from session state using the group_display_name as key
                         current_selection = st.session_state.confirmed_column_mappings.get(group_display_name, {}).get(file_name, "-- Select Header --")

                         # Ensure current_selection is in options, otherwise default to "-- Select Header --"
                         if current_selection not in options:
                              current_selection = "-- Select Header --"

                         with expander_cols[i]:
                              selected_header = st.selectbox(
                                   f"{file_name}",
                                   options=options,
                                   index=options.index(current_selection) if current_selection in options else 0,
                                   key=f"select_{group_index}_{file_name}"
                              )

                              # Update session state directly based on the selectbox value
                              # Ensure the group key exists before adding file mapping
                              if group_display_name not in st.session_state.confirmed_column_mappings:
                                   st.session_state.confirmed_column_mappings[group_display_name] = {}

                              if selected_header != "-- Select Header --":
                                   st.session_state.confirmed_column_mappings[group_display_name][file_name] = selected_header
                              else:
                                   # If "None" is selected, ensure the entry for this file is removed
                                   # Corrected indentation for this else block
                                   if file_name in st.session_state.confirmed_column_mappings.get(group_display_name, {}):
                                        del st.session_state.confirmed_column_mappings[group_display_name][file_name]
                                        # If the group is now empty in mappings, remove the group key
                                        if not st.session_state.confirmed_column_mappings.get(group_display_name, {}):
                                             # Check if the key exists before deleting to prevent KeyError
                                             if group_display_name in st.session_state.confirmed_column_mappings:
                                                  del st.session_state.confirmed_column_mappings[group_display_name]



                    # Optional: Display comparison scores within the expander
                    if st.checkbox("Show detailed comparison scores for this group", key=f"show_scores_{group_index}"):
                         if 'header_comparisons' in st.session_state:
                              st.subheader("Comparison Scores within this Group")
                              group_headers_list = [item['header'] for item in group]
                              group_comparisons = [comp for comp in st.session_state.header_comparisons
                                                    if comp['header1'] in group_headers_list and
                                                       comp['header2'] in group_headers_list and
                                                       comp['header1'] != comp['header2']]

                              if group_comparisons:
                                   comparison_df = pd.DataFrame(group_comparisons)
                                   st.dataframe(comparison_df)
                              else:
                                   st.info("No comparison data available for headers within this group.")
                         else:
                              st.info("Run the grouping first to see comparison scores.")


        st.markdown("---")
        st.subheader("Confirmed Column Mappings Summary")
        # Display the current confirmed mappings for review
        if st.session_state.confirmed_column_mappings:
            # Display a summary outside expanders
            st.write("Summary of selected headers for each group and file:")
            summary_data = []
            # Iterate through grouped_headers to maintain order and structure
            for group in st.session_state.grouped_header_matches:
                 if group:
                      group_display_name = group[0]['header'] # Use the first header as the group key
                      if group_display_name in st.session_state.confirmed_column_mappings:
                           mappings = st.session_state.confirmed_column_mappings[group_display_name]
                           for file_name, header in mappings.items():
                                summary_data.append({'Group Name': group_display_name, 'Source File': file_name, 'Selected Header': header})

            if summary_data:
                 st.dataframe(pd.DataFrame(summary_data))
            else:
                 st.info("No column mappings confirmed yet.")


            st.info("These are the mappings based on your selections above. They will be used in the next step.")
        else:
            st.info("No column mappings confirmed yet. Make selections above to confirm.")

# Note: You will need to rerun the Streamlit app cell (cell ac3251a2) after
# updating matching.py for the changes to take effect in the running app.
