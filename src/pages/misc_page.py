import streamlit as st
from streamlit_extras import colored_header
from pages.page_utility import calculate_module_import_time
import pandas as pd

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with st.spinner(f"Loading modules for ```misc_page.py``` | Approximate loading time: ```{round(calculate_module_import_time(['summarizer', 'synonym']), 4)}``` seconds"):
    load_text = st.empty()
    load_text.write("Loading ```summarizer.py```")
    from summarizer import summarize_text
    load_text.write("Loading ```synonym.py```")
    from synonym import get_synonyms
    load_text.empty()

colored_header.colored_header("Misc Tools", "Advanced synonyms & summarization", st.session_state["header_color"] if "header_color" in st.session_state else "blue-70")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=1):
        st.subheader("Find Synonyms")
        word_input = st.text_input("Enter a word")
        pos_filter = st.selectbox("Filter by Part of Speech", ["N/A", "noun", "verb", "adjective"])
        deep_search = st.checkbox("Deep Search")
        if st.button("Get Synonyms"):
            with st.spinner("Searching for synonyms..."):
                if word_input.strip():
                    synonyms_list = get_synonyms(word_input, pos_filter if pos_filter != "N/A" else None, deep_search)
                    if synonyms_list:
                        df = pd.DataFrame(synonyms_list).drop_duplicates().sort_values("word").reset_index(drop=True)
                        st.data_editor(df, use_container_width=True)
                    else:
                        st.info("No synonyms found.")
                else:
                    st.error("Please enter a word.")

with col2:
    with st.container(border=1):
        st.subheader("Summarize Text")
        text_input = st.text_area("Enter text", height=200)
        if st.button("Summarize"):
            with st.spinner("Summarizing text..."):
                if not text_input.strip():
                    st.error("Please enter some text.")
                elif len(text_input) > 4096:
                    st.error("Text exceeds 4096 character limit.")
                else:
                    summary = summarize_text(text_input)
                    st.write(summary)
