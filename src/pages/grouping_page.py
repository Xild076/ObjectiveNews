import streamlit as st
from streamlit_extras import colored_header
from pages.page_utility import calculate_module_import_time, make_notification
import sys
import os
from streamlit_extras.button_selector import button_selector
from streamlit_extras.stoggle import stoggle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with st.spinner(f"Loading modules for ```grouping_page.py``` | Approximate loading time: ```{round(calculate_module_import_time(['grouping', 'utility']), 4)}``` seconds"):
    load_text = st.empty()
    load_text.write("Loading ```utility.py```")
    from utility import normalize_text, fix_space_newline, SentenceHolder
    load_text.write("Loading ```grouping.py```")
    from grouping import observe_best_cluster
    load_text.write("Loading ```nltk```")
    import nltk
    nltk.download('punkt_tab')
    load_text.write("Loading ```sklearn.cluster```")
    from sklearn.cluster import AgglomerativeClustering, KMeans
    load_text.write("Loading ```keybert```")
    from keybert import KeyBERT
    load_text.write("Loading ```collections```")
    from collections import Counter
    load_text.empty()

colored_header.colored_header(
    "Grouping Text",
    "Group sentences by similarity of topic and relevancy.",
    st.session_state["header_color"] if "header_color" in st.session_state else "blue-70"
)
user_input = st.text_area("Enter text to group", height=400)

with st.expander("⚙️ - Settings - Set configurations for grouping"):
    max_clusters = st.slider("Maximum number of clusters", min_value=2, max_value=40, value=8)
    stoggle("ⓘ What is Maximum Number of Clusters?", "The maximum number of clusters determines the maximum number of groups the sentences will be divided into. 10 is recommended for most cases.")
    st.write("---")
    context = st.checkbox("Include context", True)
    stoggle("ⓘ What is Context?", "Context includes the surrounding sentences of the representative sentence in the cluster. This helps the algorithm understand sentences, especially short but important ones, better.")
    st.write("---")
    context_len = st.slider("Context length", min_value=1, max_value=5, value=1)
    stoggle("ⓘ What is Context Length?", "Context length determines the number of sentences to include before and after the representative sentence in the cluster.")
    st.write("---")
    context_weight = st.slider("Context weight", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    stoggle("ⓘ What is Context Weight?", "Context weight determines the importance of context in the clustering process. A higher weight means more importance goes to context.")
    st.write("---")
    preprocess = st.checkbox("Preprocess", True)
    stoggle("ⓘ What is Preprocess?", "Preprocessing includes removing stopwords and punctuation, and lemmatizing the text. This helps in better clustering by removing all unnecessary items.")
    st.write("---")
    attention = st.checkbox("Use attention", True)
    stoggle("ⓘ What is Attention?", "Attention helps in understanding the importance of each word in the sentence. This helps in better clustering by understanding the importance of each word.")
    st.write("---")
    clustering_method = button_selector(label="Clustering method", options=["KMeans", "Agglomerative"], index=0)
    stoggle("ⓘ What is Clustering Method?", "KMeans is a centroid-based clustering method that divides the data into clusters based on the distance from the centroid. Agglomerative is a hierarchical clustering method that merges the closest clusters together.")
    st.write("---")
    sil_weight = st.slider("Silhouette weight", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
    stoggle("ⓘ What is Silhouette Weight?", "Silhouette weight determines the importance of the silhouette score in the clustering process. A higher weight means more importance goes to silhouette score.")
    st.write("---")
    db_weight = st.slider("Davies Bouldin weight", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    stoggle("ⓘ What is Davies Bouldin Weight?", "Davies Bouldin weight determines the importance of the Davies Bouldin score in the clustering process. A higher weight means more importance goes to Davies Bouldin score.")
    st.write("---")
    ch_weight = st.slider("Calinski Harabasz weight", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    stoggle("ⓘ What is Calinski Harabasz Weight?", "Calinski Harabasz weight determines the importance of the Calinski Harabasz score in the clustering process. A higher weight means more importance goes to Calinski Harabasz score.")

def disable_group_button():
    st.session_state["grouping_disabled"] = True

if "grouping_disabled" not in st.session_state:
    st.session_state["grouping_disabled"] = False

submit_button = st.button("Group Text", disabled=st.session_state["grouping_disabled"], on_click=disable_group_button)

if submit_button:
    if "push_notification" not in st.session_state or st.session_state["push_notifications"]:
        notif_text = st.info("The process is underway! We will notify you once it is complete, so you can tab off and come back later. You can disable these notifications in the settings.")
    progress_bar = st.progress(0)
    load_text = st.empty()

    if user_input.strip() == "":
        st.error("Please enter some text.")
        st.session_state["grouping_disabled"] = False
        st.stop()
    
    load_text.write("Normalizing text...")
    original_text = normalize_text(user_input)
    progress_bar.progress(5)
    
    load_text.write("Extracting title keywords...")
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(
        original_text,
        keyphrase_ngram_range=(1,3),
        stop_words='english',
        top_n=1
    )
    keywords = keywords[0][0].title()
    progress_bar.progress(15)

    load_text.write("Tokenizing sentences...")
    sentences = nltk.sent_tokenize(original_text)
    sentences = [SentenceHolder(text=sent) for sent in sentences]
    progress_bar.progress(20)

    load_text.write("Grouping sentences...")
    clustering_method_class = [KMeans, AgglomerativeClustering][clustering_method]
    try:
        full_clusters_data = observe_best_cluster(sentences, max_clusters=max_clusters, 
                                        context=context, context_len=context_len,
                                        weights={'single':1-context_weight, 'context':context_weight},
                                        preprocess=preprocess, attention=attention, 
                                        clustering_method=clustering_method_class,
                                        score_weights={'sil':sil_weight, 'db':db_weight, 'ch':ch_weight})
    except:
        st.error("Please enter a longer text. There is not enough text to cluster!")
        st.session_state["grouping_disabled"] = False
        st.stop()
    clusters = full_clusters_data['clusters']
    metrics = full_clusters_data['metrics']
    progress_bar.progress(60)

    load_text.write("Extracting keywords for each cluster title...")
    visual_keywords = []
    for cluster in clusters:
        keywords_bert = kw_model.extract_keywords(
            cluster["representative_with_context"].text, keyphrase_ngram_range=(1, 3), stop_words="english", top_n=15
        )
        visual_keywords.append([kw[0] for kw in keywords_bert])
    all_keywords = [kw for cluster_kw in visual_keywords for kw in cluster_kw]
    keyword_counts = Counter(all_keywords)
    unique_cluster_keywords = []
    for i, cluster_kw in enumerate(visual_keywords):
        filtered_keywords = [kw for kw in cluster_kw if keyword_counts[kw] == 1]
        if filtered_keywords:
            unique_cluster_keywords.append(filtered_keywords[0])
        else:
            unique_cluster_keywords.append(visual_keywords[i][0])
    progress_bar.progress(100)
    load_text.empty()

    with st.container():
        st.write(f"#### Analysis Results about {keywords}...")
        for i, cluster in enumerate(clusters):
            with st.expander(f"Cluster regarding **{unique_cluster_keywords[i].title()}**...", i == 0):
                st.write("##### Representative Sentence:")
                st.write(fix_space_newline(cluster['representative'].text))
                stoggle("Representative Sentence with Context", fix_space_newline(cluster['representative_with_context'].text))

                stoggle_text = "<ul>"
                for sentence in cluster["sentences"]:
                    stoggle_text += (
                        f"<li>{fix_space_newline(sentence.text)}</li>"
                    )
                stoggle_text += "</ul>"
                stoggle("All Sentences (Click to Expand)", stoggle_text.strip())

        st.write("##### Metrics:")
        st.write(f"Silhouette Score: ```{round(metrics['silhouette'], 5)}```")
        st.write(f"Davies Bouldin Score: ```{round(metrics['davies_bouldin'], 5)}```")
        st.write(f"Calinski Harabasz Score: ```{round(metrics['calinski_harabasz'], 5)}```")
        st.write(f"Overall Score: ```{round(metrics['score'], 5)}```")
    
    progress_bar.empty()

    if "push_notification" not in st.session_state or st.session_state["push_notifications"]:
        make_notification(title="Text Grouping Complete", body="The text grouping process has been completed! You can now view the results.")
        notif_text.empty()
    st.session_state["grouping_disabled"] = False