import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stoggle import stoggle

st.set_page_config(page_title="Article Analysis", layout="centered", initial_sidebar_state="auto")

with st.spinner("Loading article modules..."):
    import nltk
    from util import validate_and_normalize_link
    from article_analysis import cluster_articles, provide_metrics, organize_clusters
    import validators
    import requests
    from text_fixer import clean_text
    nltk.download('wordnet')
    nltk.download('omw-1.4')

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

colored_header(label="Article Analysis", description="Find the key points of the news and find how objective it is.", color_name="light-blue-70")

link_input = st.text_area("Enter an article link to analyse...", placeholder="Enter a link...")
date_option = st.radio("Select which type of link was uploaded:", ("News", "Data"))
user_number = st.number_input("Enter the number of articles to fetch:", min_value=1, max_value=20, value=5, step=1)
submit_button = st.button('Analyze')

if 'organized_clusters' not in st.session_state:
    st.session_state['organized_clusters'] = None

if submit_button:
    try:
        link = link_input.strip()
        if not validators.url(link):
            st.error("Please enter a valid link.")
            st.stop()
        progress_placeholder = st.empty()
        status_text = st.empty()
        with st.spinner("Processing..."):
            progress_bar = progress_placeholder.progress(0)
            status_text.markdown("Step 1: Clustering article information...")
            clustered_articles = cluster_articles(link, date_option.lower(), user_number)
            progress_bar.progress(50)
            status_text.markdown("Step 2: Calculating metrics...")
            clustered_articles_with_metrics = provide_metrics(clustered_articles)
            progress_bar.progress(75)
            status_text.markdown("Step 3: Organizing the clusters...")
            organized_clusters = organize_clusters(clustered_articles_with_metrics)
            progress_bar.progress(100)
        progress_placeholder.empty()
        status_text.empty()
        st.session_state['organized_clusters'] = organized_clusters
    except Exception as e:
        st.error(f"An error {e} occured.")
        st.stop()

if st.session_state['organized_clusters']:
    clusters = st.session_state['organized_clusters'].copy()
    clusters = sorted(clusters, key=lambda c: (min(1 / c['reliability'] * 200, 25)) + len(c['summary']))
    clusters.reverse()

    def get_reliability_text(r):
        if r <= 15:
            return "Very Reliable"
        elif r <= 25:
            return "Reliable"
        else:
            return "Unreliable"

    for entry in clusters:
        with st.expander(entry['summary'][:70] + "..." if len(entry['summary']) > 70 else entry['summary'], expanded=False):
            st.subheader("Summary")
            st.write(clean_text(entry['summary']))
            reliability = round(entry['reliability'])
            reliability_text = get_reliability_text(reliability)
            bar_color = "linear-gradient(to right, #4CAF50, #FFEB3B, #F44336);"
            meter_html = f"""
            <div style="position: relative; width: 100%; margin: 20px 0;">
                <div style="position: relative; height: 20px; background: {bar_color}; border-radius: 10px; overflow: hidden;">
                    <div style="position: absolute; left: {reliability}%; transform: translateX(-50%); width: 2px; height: 20px; background: #000;"></div>
                </div>
                <div style="position: absolute; left: {reliability}%; transform: translateX(-50%) translateY(-100%); background: #000; color: #fff; padding: 2px 5px; font-size: 12px; border-radius: 3px; white-space: nowrap;">
                    {reliability}%
                </div>
            </div>
            <p style="text-align:center; font-size:20px; margin-top:-10px;">{reliability_text}</p>
            """
            st.subheader("Reliability")
            st.markdown(meter_html, unsafe_allow_html=True)
            st.subheader("Sources")
            source_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
            for source in entry['sources']:
                source_html += f"<span style='background:#f0f0f0; padding:5px 10px; border-radius:5px; font-size:13px;'>{source}</span>"
            source_html += "</div>"
            st.markdown(source_html, unsafe_allow_html=True)
            st.subheader("All Sentences")
            sentence_html = "<ul>"
            for sentence in entry['sentences']:
                sentence_html += f"<li>{sentence}</li>"
            sentence_html += "</ul>"
            stoggle("View Sentences", sentence_html)

st.markdown("---")

colored_header(label="Feedback", description="This article analyzer is still in its beta, being continuously updated to make it better.", color_name="light-blue-70")

feedback_input = st.text_area("Your feedback:", height=100, placeholder="Type your feedback here...")
feedback_button = st.button("Send Feedback")

if feedback_button and feedback_input.strip():
    owner = 'Xild076'
    repo = 'ObjectiveNews'
    with open('secrets/github_token.txt', 'r') as f:
        token = f.read()
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {"Authorization": f"token {token}"}
    data = {"title": "Feedback: article_page.py", "body": feedback_input}
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 201:
            st.success("Thank you for your feedback! A new GitHub issue has been created.")
        else:
            st.error("There was an error creating an issue on GitHub. Please try again later.")
    except Exception:
        st.error("There was an error sending your feedback. Please try again later.")