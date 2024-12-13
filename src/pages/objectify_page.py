import streamlit as st
from streamlit_extras.colored_header import colored_header

st.set_page_config(page_title="Objective Text", layout="centered", initial_sidebar_state="auto")

with st.spinner("Loading objectify modules..."):
    import difflib
    from textblob import TextBlob
    import objectify_text
    import text_fixer
    import time
    import requests
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def generate_diff_html(original, modified, score_original, score_modified):
    diff = list(difflib.ndiff(original.split(), modified.split()))
    original_html = []
    modified_html = []
    i = 0
    while i < len(diff):
        word = diff[i]
        if word.startswith('  '):
            original_html.append(word[2:])
            modified_html.append(word[2:])
            i += 1
        elif word.startswith('- '):
            if (i + 1) < len(diff) and diff[i + 1].startswith('+ '):
                removed_word = word[2:]
                added_word = diff[i + 1][2:]
                original_html.append(f'<span style="color:#FF4B4B; text-decoration: line-through;">{removed_word}</span>')
                modified_html.append(f'<span style="color:#4B79FF;">{added_word}</span>')
                i += 2
            else:
                removed_word = word[2:]
                original_html.append(f'<span style="color:#FF4B4B; text-decoration: line-through;">{removed_word}</span>')
                i += 1
        elif word.startswith('+ '):
            added_word = word[2:]
            modified_html.append(f'<span style="color:#4B79FF;">{added_word}</span>')
            i += 1
        else:
            i += 1
    original_text = ' '.join(original_html)
    modified_text = ' '.join(modified_html)
    objectivity_original = 100 - score_original * 100
    objectivity_modified = 100 - score_modified * 100
    delta = objectivity_modified - objectivity_original
    if delta > 0:
        improvement = f'Objectivity improved by <span style="color:#28a745;">{delta:.2f}%</span>'
    elif delta < 0:
        improvement = f'Objectivity decreased by <span style="color:#dc3545;">{abs(delta):.2f}%</span>'
    else:
        improvement = 'No change in objectivity.'
    html = f"""
    <div style="display: flex; justify-content: space-between; margin-top: 20px;">
        <div style="width: 48%; padding: 20px; background-color: #ffffff; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h3 style="color:#333; text-align: center;">Original Text</h3>
            <p style="font-size: 18px; color:#555;">{original_text}</p>
            <h4 style="color:#333; text-align: center;">Objectivity Score:</h4>
            <p style="font-size: 24px; font-weight: bold; text-align: center;">{objectivity_original:.2f}/100</p>
        </div>
        <div style="width: 48%; padding: 20px; background-color: #ffffff; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h3 style="color:#333; text-align: center;">Modified Text</h3>
            <p style="font-size: 18px; color:#555;">{modified_text}</p>
            <h4 style="color:#333; text-align: center;">Objectivity Score:</h4>
            <p style="font-size: 24px; font-weight: bold; text-align: center;">{objectivity_modified:.2f}/100</p>
        </div>
    </div>
    <div style="text-align: center; margin-top: 30px;">
        <h3>{improvement}</h3>
        <p style="font-size: 12px; color:#777;">
            Objectivity score is derived from 
            <a href="https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis" style="color:#4B79FF; text-decoration: none;">TextBlob Sentiment Analysis</a>.
        </p>
    </div>
    """
    return html

colored_header(label="Objective Text", description="Transform your text to be more objective.", color_name="light-blue-70")

user_input = st.text_area("Enter the text you want to objectify here:", height=200, placeholder="Type your text here...")
submit_button = st.button(label='Submit')

if submit_button:
    progress_placeholder = st.empty()
    status_text = st.empty()
    with st.spinner("Processing..."):
        progress_bar = progress_placeholder.progress(0)
        status_text.markdown("Step 1: Cleaning text...")
        clean_text = text_fixer.clean_text(user_input)
        time.sleep(0.5)
        progress_bar.progress(25)
        status_text.markdown("Step 2: Objectifying the text...")
        modified_text = objectify_text.objectify_text(clean_text)
        progress_bar.progress(50)
        status_text.markdown("Step 3: Preparing the display...")
        time.sleep(0.5)
        progress_bar.progress(75)
        score_original = TextBlob(user_input).subjectivity
        score_modified = TextBlob(modified_text).subjectivity
        diff_html = generate_diff_html(user_input, modified_text, score_original, score_modified)
        st.session_state["diff_html"] = diff_html
        progress_bar.progress(100)
        status_text.markdown("<p style='color:#333;'>Completed!</p>", unsafe_allow_html=True)
    time.sleep(0.5)
    progress_placeholder.empty()
    status_text.empty()

if "diff_html" in st.session_state:
    st.markdown(st.session_state["diff_html"], unsafe_allow_html=True)

st.markdown("---")

colored_header(label="Feedback", description="This objectifier is still in its beta, being continuously updated to make it better.", color_name="light-blue-70")

feedback_input = st.text_area("Your feedback:", height=100, placeholder="Type your feedback here...")
feedback_button = st.button("Send Feedback")

if feedback_button and feedback_input.strip():
    owner = 'Xild076'
    repo = 'ObjectiveNews'
    with open('secrets/github_token.txt', 'r') as f:
        token = f.read()
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {"Authorization": f"token {token}"}
    data = {"title": "Feedback: objectify_page.py", "body": feedback_input}
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 201:
            st.success("Thank you for your feedback! A new GitHub issue has been created.")
        else:
            st.error("There was an error creating an issue on GitHub. Please try again later.")
    except Exception:
        st.error("There was an error sending your feedback. Please try again later.")