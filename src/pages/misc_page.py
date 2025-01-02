import streamlit as st
from streamlit_extras import colored_header
from pages.page_utility import calculate_module_import_time, make_notification
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
    load_text.write("Loading ```utility.py```")
    from utility import normalize_text, fix_space_newline
    load_text.empty()

colored_header.colored_header("Misc Tools", "Advanced synonyms & summarization", st.session_state["header_color"] if "header_color" in st.session_state else "blue-70")

def disable_synonyms():
    st.session_state["synonyms_disabled"] = True

def disable_summarize():
    st.session_state["summarize_disabled"] = True

if "synonyms_disabled" not in st.session_state:
    st.session_state["synonyms_disabled"] = False

if "summarize_disabled" not in st.session_state:
    st.session_state["summarize_disabled"] = False

col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.subheader("Find Synonyms")
        word_input = st.text_input("Enter a word")
        pos_filter = st.selectbox("Filter by Part of Speech", ["N/A", "noun", "verb", "adjective"])
        deep_search = st.checkbox("Deep Search")
        synonyms_button = st.button("Get Synonyms", disabled=st.session_state["synonyms_disabled"], on_click=disable_synonyms)
        if synonyms_button:
            if "push_notification" not in st.session_state or st.session_state["push_notifications"]:
                notif_text = st.info("The process is underway! We will notify you once it is complete, so you can tab off and come back later. You can disable these notifications in the settings.")
            progress_bar = st.progress(0)
            load_text = st.empty()
            load_text.write("Searching for synonyms...")
            if word_input.strip():
                synonyms_list = get_synonyms(word_input, pos_filter if pos_filter != "N/A" else None, deep_search)
                if synonyms_list:
                    df = pd.DataFrame(synonyms_list).drop_duplicates().sort_values("word").reset_index(drop=True)
                    st.data_editor(df, use_container_width=True)
                else:
                    st.info("No synonyms found.")
                progress_bar.progress(100)
            else:
                st.error("Please enter a word.")
                st.session_state["synonyms_disabled"] = False
            load_text.empty()
            progress_bar.empty()
            if "push_notification" not in st.session_state or st.session_state["push_notifications"]:
                make_notification(title="Synonyms Retrieval Complete", body="The synonym search process has been completed!")
                notif_text.empty()
            st.session_state["synonyms_disabled"] = False

with col2:
    with st.container():
        st.subheader("Summarize Text")
        text_input = st.text_area("Enter text", height=200)
        summarize_button = st.button("Summarize", disabled=st.session_state["summarize_disabled"], on_click=disable_summarize)
        if summarize_button:
            if "push_notification" not in st.session_state or st.session_state["push_notifications"]:
                notif_text = st.info("The process is underway! We will notify you once it is complete, so you can tab off and come back later. You can disable these notifications in the settings.")
            progress_bar = st.progress(0)
            load_text = st.empty()
            load_text.write("Summarizing text...")
            if not text_input.strip():
                st.error("Please enter some text.")
                st.session_state["summarize_disabled"] = False
            elif len(text_input) > 4096:
                st.error("Text exceeds 4096 character limit.")
                st.session_state["summarize_disabled"] = False
            else:
                summary = normalize_text(fix_space_newline(summarize_text(text_input)))
                st.write(summary)
                progress_bar.progress(100)
            load_text.empty()
            progress_bar.empty()
            if "push_notification" not in st.session_state or st.session_state["push_notifications"]:
                make_notification(title="Text Summarization Complete", body="The text summarization process has been completed!")
                notif_text.empty()
            st.session_state["summarize_disabled"] = False