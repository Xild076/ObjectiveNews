import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stoggle import stoggle

st.set_page_config(page_title="Mini Tools", layout="wide", initial_sidebar_state="auto")

with st.spinner("Loading tools..."):
    from objectify_text import calc_objectivity_word
    from summarizer import summarize_text
    import nltk
    import requests
    import math
    import pandas as pd
    nltk.download('punkt')

st.markdown("""
<style>
textarea {
    background: #FAFAFA;
    border: 1px solid #E0E0E0;
    border-radius: 5px;
    padding:10px;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

colored_header(label="Mini Tools", description="Find synonyms sorted by objectivity and summarize text.", color_name="light-blue-70")

if 'synonyms_df' not in st.session_state:
    st.session_state['synonyms_df'] = None

if 'summary_text' not in st.session_state:
    st.session_state['summary_text'] = ""

col1, col2 = st.columns(2)

with col1:
    with st.container(border=1):
        st.subheader("Find Synonyms")
        word_input = st.text_input("Enter a word:", placeholder="Type a word...")
        find_synonyms_button = st.button("Find Synonyms", key="find_synonyms")
        if find_synonyms_button:
            if word_input.strip():
                try:
                    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word_input}"
                    response = requests.get(url)
                    data = response.json()
                    synonyms_list = []
                    if isinstance(data, list):
                        for entry in data:
                            meanings = entry.get('meanings', [])
                            for meaning in meanings:
                                part_of_speech = meaning.get('partOfSpeech', 'N/A')
                                syns = meaning.get('synonyms', [])
                                for syn in syns:
                                    synonyms_list.append({'Synonym': syn, 'PartOfSpeech': part_of_speech})
                    if not synonyms_list:
                        st.info("No synonyms found.")
                        st.session_state['synonyms_df'] = None
                    else:
                        df = pd.DataFrame(synonyms_list)
                        df = df.drop_duplicates(subset=['Synonym', 'PartOfSpeech'])
                        df['Objectivity'] = df['Synonym'].apply(calc_objectivity_word)
                        df = df.sort_values(by='Objectivity', ascending=False).reset_index(drop=True)
                        st.session_state['synonyms_df'] = df
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Please enter a word.")
        if st.session_state['synonyms_df'] is not None:
            df_display = st.session_state['synonyms_df'].copy()
            df_display.index = range(1, len(df_display) + 1)
            st.dataframe(df_display, height=600, use_container_width=True, hide_index=True)

with col2:
    with st.container(border=1):
        st.subheader("Summarize Text")
        text_input = st.text_area("Enter text to summarize:", height=300, placeholder="Type or paste text here...")
        summary_option = st.select_slider("Select summary length:", options=["Short", "Medium", "Long"])
        summarize_button = st.button("Summarize", key="summarize")
        if text_input.strip():
            char_count = len(text_input)
            if summary_option == "Short":
                multiplier = 0.05
            elif summary_option == "Medium":
                multiplier = 0.1
            else:
                multiplier = 0.15
            min_length = max(math.floor(char_count * multiplier), 50)
            max_length = max(math.floor(char_count * multiplier * 2), min_length + 50)
        else:
            max_length = 200
            min_length = 150
        if summarize_button:
            if text_input.strip():
                try:
                    summary = summarize_text(text_input, min_length=min_length, max_length=max_length)
                    st.session_state['summary_text'] = summary
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Please enter text to summarize.")
        if st.session_state['summary_text']:
            st.write("### Summary")
            st.write(st.session_state['summary_text'])

st.markdown("---")

with st.container():
    colored_header(label="Feedback", description="Help us improve these tools by providing your feedback.", color_name="light-blue-70")
    feedback_input = st.text_area("Your feedback:", height=100, placeholder="Type your feedback here...")
    feedback_button = st.button("Send Feedback")
    if feedback_button and feedback_input.strip():
        owner = 'Xild076'
        repo = 'ObjectiveNews'
        try:
            with open('secrets/github_token.txt', 'r') as f:
                token = f.read().strip()
            url = f"https://api.github.com/repos/{owner}/{repo}/issues"
            headers = {"Authorization": f"token {token}"}
            data = {"title": "Feedback: mini_page.py", "body": feedback_input}
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 201:
                st.success("Thank you for your feedback! A new GitHub issue has been created.")
                st.session_state['feedback_input'] = ""
            else:
                st.error("There was an error creating an issue on GitHub. Please try again later.")
        except Exception:
            st.error("There was an error sending your feedback. Please try again later.")