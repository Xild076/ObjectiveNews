import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stoggle import stoggle

st.set_page_config(
    page_title="Grouping Text",
    layout="centered",
    initial_sidebar_state="auto"
)

with st.spinner("Loading clustering module..."):
    from grouping import cluster_text
    import numpy as np
    from typing import List, Dict, Any, Union, Type
    from sklearn.cluster import AgglomerativeClustering, KMeans
    import pandas as pd
    from nltk import sent_tokenize
    import requests

colored_header(
    label="Grouping Text",
    description="Getting the main points of the text.",
    color_name="light-blue-70"
)

if 'clusters' not in st.session_state:
    st.session_state.clusters = None

text_input = st.text_area(
    "Enter the text to group here...",
    placeholder="Enter text...",
    height=200
)

st.sidebar.header("Grouping Options")

clustering_method = st.sidebar.selectbox(
    "Grouping Method",
    options=["AgglomerativeClustering", "KMeans"],
    help="Choose the grouping algorithm to use."
)

st.sidebar.subheader("Score Weights")
silhouette_weight = st.sidebar.slider(
    "Silhouette Score Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.55,
    step=0.05,
    help="Weight for the Silhouette score."
)
dbscan_weight = st.sidebar.slider(
    "DB Score Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="Weight for the DB score."
)
calinski_weight = st.sidebar.slider(
    "Calinski-Harabasz Score Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05,
    help="Weight for the Calinski-Harabasz score."
)
score_weights = {
    'sil': silhouette_weight,
    'db': dbscan_weight,
    'ch': calinski_weight
}

lemmatize = st.sidebar.checkbox(
    "Lemmatize Text",
    value=True,
    help="Enable lemmatization of the text."
)

context = st.sidebar.checkbox(
    "Use Context",
    value=False,
    help="Enable contextual information in clustering."
)

if context:
    st.sidebar.subheader("Context Weights")
    single_weight = st.sidebar.slider(
        "Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Weight for the individual sentence."
    )
    context_weights = {
        'single': single_weight,
        'context':  1 - single_weight
    }
else:
    context_weights = {'single': 0.5, 'context': 0.5} 

group_button = st.button("Group Text")

if group_button:
    if not text_input.strip():
        st.warning("Please enter some text to group.")
    else:
        with st.spinner("Clustering the text..."):
            sentences = sent_tokenize(text_input.strip())
            
            if clustering_method == "AgglomerativeClustering":
                clustering_cls = AgglomerativeClustering
            elif clustering_method == "KMeans":
                clustering_cls = KMeans
            else:
                st.error("Unsupported clustering method selected.")
                st.stop()
            
            try:
                max_clusters = max(round(len(sentences) / 6), 15)
                clusters_result = cluster_text(
                    sentences=sentences,
                    clustering_method=clustering_cls,
                    context_weights=context_weights if context else {'single': 0.5, 'context': 0.5},
                    score_weights=score_weights,
                    context=context,
                    lemmatize=lemmatize,
                    max_clusters=max_clusters
                )
                
                clusters = clusters_result['clusters']
                st.session_state.clusters = clusters
            except Exception as e:
                st.error(f"An error occurred during clustering: {e}")
                st.session_state.clusters = None

if st.session_state.clusters:
    for cluster in st.session_state.clusters:
        with st.expander(cluster['representative'][:70] + "..." if len(cluster['representative']) > 70 else cluster['representative'], expanded=False):
            st.subheader("Representative Sentence")
            st.write(cluster['representative'])
            st.markdown("**With context:**")
            st.write(cluster['representative_with_context'])
            st.subheader("All Sentences")
            sentence_html = "<ul>"
            for sentence in cluster['sentences']:
                sentence_html += f"<li>{sentence.strip()}</li>"
            sentence_html += "</ul>"
            stoggle("View Sentences", sentence_html)

st.markdown("---")

colored_header(label="Feedback", description="This article analyzer is still in its beta, being continuously updated to make it better.", color_name="light-blue-70")

feedback_input = st.text_area("Your feedback:", height=100, placeholder="Type your feedback here...")
feedback_button = st.button("Send Feedback")

if feedback_button and feedback_input.strip():
    owner = 'Xild076'
    repo = 'ObjectiveNews'
    try:
        with open('secrets/github_token.txt', 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        st.error("GitHub token not found. Please ensure the 'secrets/github_token.txt' file exists.")
        token = None

    if token:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        headers = {"Authorization": f"token {token}"}
        data = {"title": "Feedback: grouping_page.py", "body": feedback_input}
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 201:
                st.success("Thank you for your feedback! A new GitHub issue has been created.")
            else:
                st.error(f"There was an error creating an issue on GitHub. Status Code: {response.status_code}")
        except Exception as e:
            st.error(f"There was an error sending your feedback: {e}")