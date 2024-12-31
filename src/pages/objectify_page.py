import streamlit as st
from streamlit_extras import colored_header
from pages.page_utility import calculate_module_import_time, estimate_time_taken_objectify, make_notification
import sys
import os
from streamlit_extras.button_selector import button_selector
from streamlit_extras.stoggle import stoggle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with st.spinner(f"Loading modules for ```objectify_page.py``` | Approximate loading time: ```{round(calculate_module_import_time(['objectify', 'utility']), 4)}``` seconds"):
    load_text = st.empty()
    load_text.write("Loading ```utility.py```")
    from utility import normalize_text
    load_text.write("Loading ```objectify.py```")
    import objectify
    load_text.write("Loading ```textblob```")
    from textblob import TextBlob
    load_text.write("Loading ```keybert```")
    from keybert import KeyBERT
    load_text.empty()

colored_header.colored_header(
    "Objectify Text",
    "This tool makes text less subjective and more objective by removing or replacing subjective words.",
    st.session_state["header_color"] if "header_color" in st.session_state else "blue-70"
)

user_input = st.text_area(
    "Enter the text to objectify here:",
    height=300,
    placeholder="Your text here..."
)

with st.expander("⚙️ - Settings - Set configurations for objectification"):
    objectivity_threshold = st.slider(
        "Objectivity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01
    )
    stoggle(
        "ⓘ What is Objectivity Threshold?",
        "The objectivity threshold determines at what level will words be removed. The higher the threshold, the more words will be removed, with even objective descriptive words being removed/altered. The lower the threshold, the more the focus will be on more subjective words with mildly subjective words being ignored."
    )
    st.markdown("---")
    synonym_search_methodology = button_selector(
        label="Synonym Search Method",
        options=["Transformer", "WordNet"],
        index=0,
    )
    stoggle(
        "ⓘ Synonym Search Methodology",
        "Choose between 'Transformer' for advanced contextual synonyms or 'WordNet' for traditional synonym replacement. The 'Transformer' produces better contextual results but is unpredictable and sometimes produces non-synonyms. The 'WordNet' method is more reliable but may not always fit into the context as well as the 'Transformer' method."
    )

submit_button = st.button("Objectify")

def diff_text(old_text, new_text):
    import difflib
    old_tokens = old_text.split()
    new_tokens = new_text.split()
    diff = list(difflib.ndiff(old_tokens, new_tokens))
    result = []
    for token in diff:
        if token.startswith('- '):
            result.append(f"<span style='color:red;text-decoration:line-through'>{token[2:]}</span>")
        elif token.startswith('+ '):
            result.append(f"<span style='color:blue'>{token[2:]}</span>")
        elif token.startswith('  '):
            result.append(token[2:])
    return " ".join(result)

def escape_for_js(text):
    return (
        text.replace("\\", "\\\\")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace('"', '\\"')
            .replace("'", "\\'")
    )

if submit_button:
    if "push_notification" not in st.session_state or st.session_state["push_notifications"]:
        notif_text = st.info("The process is underway! We will notify you once it is complete, so you can tab off and come back later. You can disable these notifications in the settings.")
    progress_bar = st.progress(0)
    load_time = st.empty()
    load_time.write(f"Estimated time taken to objectify: ```{round(estimate_time_taken_objectify(user_input), 4)}``` seconds")
    original_text = normalize_text(user_input)
    progress_bar.progress(10)
    if original_text.strip() == "":
        st.error("Please enter some text to objectify.")
        st.stop()
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(
        original_text,
        keyphrase_ngram_range=(1,3),
        stop_words='english',
        top_n=1
    )
    keywords = keywords[0][0].title()
    progress_bar.progress(20)
    objectified_text = objectify.objectify_text(
        original_text,
        objectivity_threshold=objectivity_threshold,
        synonym_search_methodology=['transformer', 'wordnet'][synonym_search_methodology]
    )
    progress_bar.progress(90)
    blob_old = TextBlob(original_text)
    blob_new = TextBlob(objectified_text)
    old_obj = 1 - blob_old.sentiment.subjectivity
    new_obj = 1 - blob_new.sentiment.subjectivity
    progress_bar.progress(100)
    with st.container():
        st.write(f"#### Objectified text about {keywords}...")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="font-size: 18px; text-align: center;">
                <b>Old Objectivity:</b> {round(old_obj * 100, 5)} → <br>
                <b>New Objectivity:</b> {round(new_obj * 100, 5)}
            </div>
            """, unsafe_allow_html=True)
        with col2:
            change_percentage = ((new_obj - old_obj) / old_obj) * 100 if old_obj != 0 else 0
            arrow = "↑" if change_percentage > 0 else "↓"
            color = "green" if change_percentage > 0 else "red"
            st.markdown(f"""
            <div style="font-size: 18px; text-align: center;">
                <b>Change:</b> <span style="color: {color};">
                    {abs(change_percentage):.2f}% {arrow}
                </span>
            </div>
            """, unsafe_allow_html=True)
        if old_obj < new_obj:
            st.markdown(f"""
            <div style="font-size: 18px; text-align: center;">
                <br>Congratulations! Your text has been altered to be more objective!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="font-size: 18px; text-align: center;">
                <br>Your text is already quite objective. Good job and keep up the good work!
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            with st.container(border=1):
                st.markdown("##### Original Text")
                st.markdown(diff_text(original_text, objectified_text), unsafe_allow_html=True)
        with col4:
            with st.container(border=1):
                escaped_objectified = escape_for_js(objectified_text)
                copy_button_code = f"""
                <style>
                .copy-container {{
                    position: relative;
                    margin-bottom: 10px;
                }}
                .copy-btn {{
                    background: none;
                    border: none;
                    cursor: pointer;
                    color: #31333F;
                    position: absolute;
                    top: 0;
                    right: 0;
                    font-size: 1rem;
                }}
                .copy-btn:hover {{
                    color: #4C9EE3;
                }}
                </style>
                <script>
                function copyToClipboard() {{
                    navigator.clipboard.writeText("{escaped_objectified}").then(
                        () => console.log('Copied to clipboard.'),
                        (err) => console.error('Failed to copy text: ', err)
                    );
                }}
                </script>
                <div class="copy-container">
                    <h5 style="margin: 0;">Objectified Text</h5>
                    <button class="copy-btn" onclick="copyToClipboard()">⎘ Copy</button>
                </div>
                """
                st.markdown(copy_button_code, unsafe_allow_html=True)
                st.markdown(objectified_text)
    progress_bar.empty()
    load_time.empty()

    if "push_notification" not in st.session_state or st.session_state["push_notifications"]:
        make_notification("Text Objectification", "Your text has been successfully objectified!")
        notif_text.empty()