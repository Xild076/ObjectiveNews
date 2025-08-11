def render_sentences(sentences):
    if not sentences:
        return "<div style='color:#b71c1c;font-size:1em;'>No contributing sentences were found for this narrative.</div>"
    html = ""
    for sent in sentences:
        if sent is None or not hasattr(sent, 'source') or not hasattr(sent, 'text'):
            continue
        source_domain = getattr(sent, 'source', 'N/A')
        reliability_percent = fetch_source_reliability(source_domain) * 100
        html += f"""
            <div style='background:#fff;border-radius:8px;padding:0.7em 1em;margin-bottom:0.7em;border:1px solid #e0e5ec;'>
                <div style='font-size:1.02em;color:#222;'>{getattr(sent, 'text', '')}</div>
                <div style='font-size:0.95em;color:#666;margin-top:0.3em;'>Source: <span style='color:#1976d2;'>{source_domain}</span> | Date: <span style='color:#1976d2;'>{getattr(sent, 'date', 'N/A')}</span> | Source Reliability: <span style='color:#1976d2;'>{reliability_percent:.1f}%</span></div>
            </div>
        """
    return html
import streamlit as st
import logging
from article_analysis import article_analysis
from reliability import get_source_label, normalize_domain, _load_source_df
from utility import load_keybert
import torch
import gc

st.set_page_config(page_title="Article Analysis", layout="wide")

if 'kw_model' not in st.session_state:
    st.session_state.kw_model = load_keybert()

def generate_cluster_title(_representative_text: str) -> str:
    if not _representative_text:
        return "Untitled Narrative"
    try:
        kws = st.session_state.kw_model.extract_keywords(
            _representative_text, keyphrase_ngram_range=(2, 4), stop_words='english', use_mmr=True, diversity=0.5, top_n=1
        )
        if kws:
            return kws[0][0].title()
    except Exception:
        pass
    return _representative_text[:60].strip().title() + "..."

@st.cache_data
def fetch_source_reliability(domain: str) -> float:
    if not domain or not isinstance(domain, str):
        return 0.0
    try:
        normalized = normalize_domain(domain)
        df = _load_source_df()
        score, _ = get_source_label(normalized, df)
        if score is None or not isinstance(score, (int, float)):
            return 0.0
        if 0.0 <= score <= 1.0:
            return max(0.0, min(score, 1.0))
        return max(0.0, min(score / 100.0, 1.0))
    except Exception:
        return 0.0

def display_analysis_results(analysis_result):
    st.markdown("<h2 style='margin-bottom:0.5em;color:#222;font-weight:600;'>Article Analysis Results</h2>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:1.1em;color:#555;margin-bottom:2em;'>Distinct Narratives: <span style='color:#1976d2;font-weight:700'>{len(analysis_result)}</span></div>", unsafe_allow_html=True)
    sorted_clusters = sorted(analysis_result, key=lambda x: x.get('reliability', 0), reverse=True)
    for i, cluster in enumerate(sorted_clusters):
        reliability = cluster.get('reliability', 0.0)
        rep_text = cluster.get('representative', None)
        rep_text = rep_text.text if rep_text else cluster.get('summary', '')
        title = generate_cluster_title(rep_text)
        details = cluster.get('reliability_details', {})
        details_text = " | ".join([f"{k.replace('_', ' ').title()}: {v:.1f}%" for k, v in details.items() if isinstance(v, float)])
        st.markdown(f"""
            <div style='background:linear-gradient(90deg,#f5f7fa,#c3cfe2);border-radius:12px;padding:1.5em 2em;margin-bottom:2em;box-shadow:0 2px 8px rgba(0,0,0,0.04);'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div style='flex:3;'>
                        <div style='font-size:1.3em;font-weight:600;color:#222;margin-bottom:0.2em;'>{title}</div>
                        <div style='font-size:1.05em;color:#444;margin-bottom:0.7em;'>{cluster.get('summary', 'N/A')}</div>
                    </div>
                    <div style='flex:1;text-align:right;'>
                        <div style='font-size:1.1em;color:#1976d2;font-weight:700;'>Reliability: {reliability:.1f}%</div>
                        <div style='font-size:0.95em;color:#888;'>{details_text}</div>
                        <div style='margin-top:0.5em;width:100%;height:8px;background:#e3eafc;border-radius:4px;'><div style='width:{reliability:.1f}%;height:100%;background:#1976d2;border-radius:4px;'></div></div>
                    </div>
                </div>
                <details style='margin-top:1.2em;'>
                    <summary style='font-size:1em;color:#1976d2;cursor:pointer;'>Contributing Sentences & Sources</summary>
                    {render_sentences(cluster.get('sentences', []))}
            </div>
        """, unsafe_allow_html=True)


st.title("Article Analysis")
st.markdown("Enter a topic or URL to retrieve multiple sources, identify core narratives, and see them as summarized, objective clusters with reliability scores.")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

import gc
if st.session_state.analysis_result:
    display_analysis_results(st.session_state.analysis_result)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Start New Analysis", use_container_width=True, type="primary"):
        st.session_state.analysis_result = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.rerun()
else:
    with st.form("analysis_form"):
        st.markdown("### Analyze a Topic or News URL")
        text_input = st.text_input("Topic or URL", placeholder="e.g., 'global economic outlook'", help="Enter a news topic or paste a news article URL.")
        col1, col2, col3 = st.columns(3)
        link_count = col1.slider("Articles to Fetch", 5, 20, 10, 1, help="Number of articles to analyze.")
        summarize_level = col2.select_slider("Summarization Detail", ["fast", "medium", "best"], value="medium", help="Heavier models are disabled for this app.")
        diverse_links = col3.toggle("Use Diverse Sources", value=True, help="Fetch articles from a wider variety of domains.")
        submitted = st.form_submit_button("Analyze Topic", use_container_width=True, type="primary")

    if submitted and text_input:
        gc.collect()
        progress_bar = st.progress(0, text="Starting analysis...")
        try:
            def update_progress(value, text):
                progress_bar.progress(value, text=f"Analyzing... {text}")
            with st.spinner("Processing... this may take a moment."):
                level = {"best": "slow"}.get(summarize_level, summarize_level)
                result = article_analysis(
                    text=text_input, link_n=link_count, diverse_links=diverse_links,
                    summarize_level=level, progress_callback=update_progress
                )
            st.session_state.analysis_result = result
            progress_bar.empty()
            st.rerun()
        except Exception as e:
            progress_bar.empty()
            st.error(f"An error occurred during analysis: {e}")
            logging.exception("Analysis failed.")
    elif submitted:
        st.warning("Please enter a topic or URL to begin.")