import streamlit as st
from objectify.synonym import get_synonyms
from objectify.objectify import calculate_objectivity
from reliability import get_source_label, normalize_domain, _load_source_df

st.title("Utilities Explorer")
st.markdown("Explore some of the underlying components of the analysis engine.")

tab1, tab2 = st.tabs(["Synonym Objectivity Ranker", "Domain Reliability Checker"])

with tab1:
    st.header("Synonym Objectivity Ranker")
    st.markdown("Find synonyms for a word, automatically ranked by their calculated objectivity score. A higher score (closer to 1.0) means more neutral language.")
    st.caption("Try adjectives, loaded terms, or even names.")

    word_input = st.text_input("Enter a word to analyze:", "beautiful", help="Type any word to see its synonyms ranked by objectivity.")
    if word_input:
        with st.spinner(f"Finding and ranking synonyms for '{word_input}'..."):
            synonyms_raw = get_synonyms(word_input, deep=True, include_external=True, topn=15)
            if not synonyms_raw:
                st.warning(f"No synonyms found for '{word_input}'.")
            else:
                synonym_data = sorted(
                    [{
                        "word": syn['word'],
                        "definition": syn.get('definition'),
                        "objectivity": calculate_objectivity(syn['word'])
                    } for syn in synonyms_raw],
                    key=lambda x: x['objectivity'],
                    reverse=True
                )
                
                st.markdown("---")
                for i, syn in enumerate(synonym_data):
                    with st.container(border=True):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{i+1}. {syn['word']}**")
                            if syn.get('definition'):
                                st.caption(f"_{syn['definition']}_")
                        with col2:
                            st.metric(label="Objectivity", value=f"{syn['objectivity']:.3f}", help="A score of 1.0 is completely neutral.")

with tab2:
    st.header("Domain Reliability Checker")
    st.markdown("Check the internally stored reliability score for a news domain. The score ranges from **-1** (low reliability) to **+1** (high reliability).")
    st.caption("Try domains like `cnn.com`, `foxnews.com`, `bbc.com`, etc.")

    domain_input = st.text_input("Enter a domain to check:", "nytimes.com", placeholder="e.g., cnn.com", help="Type a news domain to see its reliability score.")
    if domain_input:
        with st.container(border=True):
            normalized = normalize_domain(domain_input)
            score, _ = get_source_label(normalized, _load_source_df())
            percent = (score + 1) / 2 * 100
            
            st.subheader(f"Reliability for: `{normalized}`")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Reliability Score [-1 to 1]", value=f"{score:.3f}")
            with col2:
                st.metric(label="Reliability Percentage", value=f"{percent:.1f}%")
            
            st.progress(int(percent), text=f"{percent:.1f}%")
        st.info("This score is an internal metric influenced by the objectivity, coverage, and recency of articles processed from that domain.", icon="ℹ️")

    st.divider()
    st.header("Media Bias & Reliability Resources")
    st.markdown("The engine's initial reliability scores are based on established datasets, which are then dynamically updated. Here are some key resources for understanding media analysis:")
    st.markdown("""
    - **[FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet):** A primary data source for our initial model training.
    - **[Ad Fontes Media](https://adfontesmedia.com/):** Creator of the Media Bias Chart.
    - **[AllSides](https://www.allsides.com/):** Provides media bias ratings and stories from multiple perspectives.
    - **[Pew Research Center](https://www.pewresearch.org/journalism/):** Conducts non-partisan research on the news industry.
    """)