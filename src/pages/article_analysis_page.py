import json
import csv
import io
from collections import Counter
import streamlit as st
import logging
from utility import load_keybert, ensure_nltk_data
ensure_nltk_data()
from reliability import get_source_label, normalize_domain, _load_source_df
import torch
import gc

if 'kw_model' not in st.session_state:
    st.session_state.kw_model = load_keybert()

def generate_cluster_title(_representative_text: str) -> str:
    if not _representative_text:
        return "Untitled Narrative"
    try:
        title = _cached_title_from_text(_representative_text)
        if title:
            return title
    except Exception:
        pass
    return _representative_text[:60].strip().title() + "..."

@st.cache_data
def _cached_title_from_text(text: str) -> str:
    try:
        kws = st.session_state.kw_model.extract_keywords(
            text, keyphrase_ngram_range=(2, 4), stop_words='english', use_mmr=True, diversity=0.5, top_n=1
        )
        if kws:
            return kws[0][0].title()
    except Exception:
        return ""
    return ""

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

def sources_for_cluster(cluster):
    sents = cluster.get('sentences', []) or []
    srcs = [getattr(s, 'source', 'unknown') if not isinstance(s, tuple) else getattr(s[0], 'source', 'unknown') for s in sents]
    srcs = [s or 'unknown' for s in srcs]
    return Counter(srcs)

def to_download_blobs(clusters):
    data = []
    for idx, c in enumerate(clusters, 1):
        sents = c.get('sentences', []) or []
        rep = c.get('representative')
        rep_text = rep.text if hasattr(rep, 'text') else None
        data.append({
            'cluster_id': idx,
            'title': generate_cluster_title(rep_text or c.get('summary', '') or ''),
            'summary': c.get('summary', ''),
            'reliability': c.get('reliability', 0.0),
            'sources': list(sources_for_cluster(c).items()),
            'sentences': [
                {
                    'text': (s[0].text if isinstance(s, tuple) and hasattr(s[0], 'text') else getattr(s, 'text', '')),
                    'source': (getattr(s[0], 'source', 'unknown') if isinstance(s, tuple) else getattr(s, 'source', 'unknown')),
                    'date': (getattr(s[0], 'date', None) if isinstance(s, tuple) else getattr(s, 'date', None)),
                } for s in sents
            ]
        })
    json_blob = json.dumps(data, ensure_ascii=False, indent=2)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['cluster_id','title','reliability','summary','sentence_text','source','date'])
    for item in data:
        for s in item['sentences']:
            writer.writerow([item['cluster_id'], item['title'], f"{item['reliability']:.1f}", item['summary'], s['text'], s['source'], s['date'] or ''])
    csv_blob = output.getvalue()
    return json_blob, csv_blob

def filter_sort_clusters(clusters, min_reliability, multi_source_only, sort_by, keyword):
    def keep(c):
        if multi_source_only and len(sources_for_cluster(c)) < 2:
            return False
        if c.get('reliability', 0.0) < min_reliability:
            return False
        if keyword:
            k = keyword.lower()
            if k not in (c.get('summary','') or '').lower() and k not in (getattr(c.get('representative'), 'text', '') or '').lower():
                return False
        return True
    kept = [c for c in clusters if keep(c)]
    if sort_by == 'Reliability':
        kept.sort(key=lambda x: x.get('reliability', 0.0), reverse=True)
    elif sort_by == 'Title':
        def t(c):
            rep = c.get('representative')
            rt = rep.text if hasattr(rep, 'text') else ''
            return generate_cluster_title(rt or c.get('summary','') or '')
        kept.sort(key=lambda x: t(x))
    elif sort_by == 'Sources':
        kept.sort(key=lambda x: len(sources_for_cluster(x)), reverse=True)
    return kept

def display_analysis_results(analysis_result):
    st.subheader("Article Analysis Results")
    sources = Counter()
    for c in analysis_result:
        sources.update(sources_for_cluster(c))
    avg_rel = 0.0
    if analysis_result:
        avg_rel = sum(c.get('reliability', 0.0) for c in analysis_result) / len(analysis_result)
    m1, m2, m3 = st.columns(3)
    m1.metric("Narratives", len(analysis_result))
    m2.metric("Distinct Sources", len(sources))
    m3.metric("Avg Reliability", f"{avg_rel:.1f}%")
    with st.expander("Filters", expanded=False):
        c1, c2, c3, c4 = st.columns([1,1,1,2])
        min_rel = c1.slider("Min Reliability", 0, 100, 0, 5)
        multi_only = c2.toggle("Multi-source only", False)
        sort_by = c3.selectbox("Sort by", ["Reliability","Sources","Title"], index=0)
        keyword = c4.text_input("Search in summaries", "")
    filtered = filter_sort_clusters(analysis_result, float(min_rel), multi_only, sort_by, keyword)
    if not filtered:
        st.info("No narratives match the current filters. Showing all narratives instead.")
        filtered = filter_sort_clusters(analysis_result, 0.0, False, sort_by, keyword)
    jb, cb = to_download_blobs(filtered)
    dl1, dl2, _ = st.columns([1,1,6])
    dl1.download_button("Download JSON", data=jb, file_name="analysis.json", mime="application/json", use_container_width=True)
    dl2.download_button("Download CSV", data=cb, file_name="analysis.csv", mime="text/csv", use_container_width=True)
    st.markdown("---")
    top_k = st.slider("Max sentences per narrative", 3, 15, 5, 1)
    st.caption(f"Showing {len(filtered)} of {len(analysis_result)} narratives")
    for idx, cluster in enumerate(filtered, 1):
        rep = cluster.get('representative')
        rep_text = rep.text if hasattr(rep, 'text') else cluster.get('summary', '')
        title = generate_cluster_title(rep_text)
        reliability = cluster.get('reliability', 0.0)
        src_counts = sources_for_cluster(cluster)
        domain_rels = {d: int(fetch_source_reliability(d)*100) for d in src_counts.keys()}
        c = st.container(border=True)
        with c:
            h, r = st.columns([4,2])
            with h:
                st.markdown(f"**{idx}. {title}**")
                st.write(cluster.get('summary', ''))
            with r:
                st.metric("Reliability", f"{reliability:.1f}%")
                bar = f"<div style='width:100%;height:8px;background:#e3eafc;border-radius:4px;'><div style='width:{min(100,max(0,int(reliability)))}%;height:8px;background:#1976d2;border-radius:4px;'></div></div>"
                st.markdown(bar, unsafe_allow_html=True)
            chips = []
            for d, cnt in list(src_counts.items())[:8]:
                rel = domain_rels.get(d, 0)
                chips.append(f"{d} ({cnt}) — {rel}%")
            if chips:
                st.caption("Sources: " + ", ".join(chips))
            with st.expander("Contributing sentences"):
                shown = 0
                sents = cluster.get('sentences', []) or []
                for s in sents:
                    if shown >= top_k:
                        break
                    if isinstance(s, tuple):
                        s0 = s[0]
                    else:
                        s0 = s
                    if not hasattr(s0, 'text'):
                        continue
                    domain = getattr(s0, 'source', 'unknown') or 'unknown'
                    date = getattr(s0, 'date', 'N/A') or 'N/A'
                    relp = domain_rels.get(domain, int(fetch_source_reliability(domain)*100))
                    st.write(f"- {s0.text}")
                    st.caption(f"{domain} • {date} • Source reliability {relp}%")
                    shown += 1


st.title("Article Analysis")
st.markdown("Enter a topic or URL to retrieve multiple sources, identify core narratives, and see them as summarized, objective clusters with reliability scores.")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

import gc
if st.session_state.analysis_result:
    display_analysis_results(st.session_state.analysis_result)
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns([1,1,6])
    if cols[0].button("New Analysis", use_container_width=True, type="primary"):
        st.session_state.analysis_result = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.rerun()
else:
    with st.form("analysis_form"):
        st.markdown("### Analyze a Topic or News URL")
        default_value = st.session_state.get('prefill', '')
        text_input = st.text_input("Topic or URL", value=default_value, placeholder="e.g., 'global economic outlook'", help="Enter a news topic or paste a news article URL.")
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
                from article_analysis import article_analysis
                result = article_analysis(
                    text=text_input, link_n=link_count, diverse_links=diverse_links,
                    summarize_level=level, progress_callback=update_progress
                )
            st.session_state.analysis_result = result
            progress_bar.empty()
            st.toast("Analysis complete", icon="✅")
            st.rerun()
        except Exception as e:
            progress_bar.empty()
            st.error(f"An error occurred during analysis: {e}")
            logging.exception("Analysis failed.")
    elif submitted:
        st.warning("Please enter a topic or URL to begin.")

    st.caption("Try an example:")
    ex1, ex2, ex3, _ = st.columns([1,1,1,6])
    if ex1.button("US elections"):
        st.session_state['prefill'] = 'US elections'
        st.rerun()
    if ex2.button("AI regulation"):
        st.session_state['prefill'] = 'AI regulation'
        st.rerun()
    if ex3.button("Climate change"):
        st.session_state['prefill'] = 'Climate change'
        st.rerun()

if 'prefill' in st.session_state and not st.session_state.get('analysis_result'):
    st.info(f"Prefilled topic: {st.session_state.prefill}")