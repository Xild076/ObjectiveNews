import streamlit as st
from streamlit_extras import colored_header
from pages.page_utility import calculate_module_import_time, estimate_time_taken_article, make_notification
import sys
import os
from streamlit_extras.button_selector import button_selector
from streamlit_extras.stoggle import stoggle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with st.spinner(f"Loading modules for ```article_page.py``` | Approximate loading time: ```{round(calculate_module_import_time(['objectify', 'article_analysis']), 4)}``` seconds"):
    load_text = st.empty()
    load_text.write("Loading ```utility.py```")
    from utility import normalize_text
    load_text.write("Loading ```article_analysis.py```")
    from article_analysis import (
        is_cluster_valid,
        process_text_input_for_keyword,
        retrieve_information_online,
        group_individual_article,
        group_representative_sentences,
        calculate_reliability,
        objectify_and_summarize
    )
    load_text.write("Loading ```keybert```")
    from keybert import KeyBERT
    load_text.write("Loading ```collections```")
    from collections import Counter
    load_text.write("Loading ```dateutil```")
    from dateutil import parser
    load_text.empty()
colored_header.colored_header(
    "Article Analysis",
    "This tool allows you to analyze an article and summarize it, find the reliability of the article, and more.",
    st.session_state["header_color"] if "header_color" in st.session_state else "blue-70"
)
user_input = st.text_area(
    "Enter any information applicable to search:",
    height=100,
    placeholder="Enter links, keywords, texts, etc..."
)
def update_estimate():
    st.session_state["estimated_time"] = estimate_time_taken_article(
        st.session_state["link_number"]
    )
if "estimated_time" not in st.session_state:
    st.session_state["estimated_time"] = estimate_time_taken_article(10)
col1, col2 = st.columns(2)
with col1:
    link_number = st.slider(
        "Search Depth Limit",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        key="link_number",
        on_change=update_estimate
    )
with col2:
    st.write("Estimated Time Taken:")
    st.write(f"```{st.session_state['estimated_time']}``` seconds")
submit_button = st.button("Analyze")
if submit_button:
    if "push_notification" not in st.session_state or st.session_state["push_notifications"]:
        notif_text = st.info("The process is underway! We will notify you once it is complete, so you can tab off and come back later. You can disable these notifications in the settings.")
    progress_bar = st.progress(0)
    load_text = st.empty()
    if user_input.strip() == "":
        st.error("Please enter some information to search.")
        st.stop()
    load_text.write("Processing user input...")
    processed_text = process_text_input_for_keyword(user_input)
    keywords = processed_text["keywords"]
    extra_info = processed_text["extra_info"]
    progress_bar.progress(10)
    load_text.write("Fetching online information...")
    articles, _ = retrieve_information_online(keywords, link_num=link_number, extra_info=extra_info)
    progress_bar.progress(40)
    if extra_info:
        load_text.write("Extracting keywords from the initial article...")
        kw_model = KeyBERT()
        keywords_bert = kw_model.extract_keywords(
            extra_info["text"], keyphrase_ngram_range=(1, 1), stop_words="english", top_n=10
        )
        keywords_bert = [kw[0] for kw in keywords_bert]
        progress_bar.progress(45)
    else:
        keywords_bert = None
    load_text.write("Grouping individual articles...")
    rep_sentences = []
    for article in articles:
        rep_sentences.extend(group_individual_article(article))
    del articles
    import gc
    gc.collect()
    progress_bar.progress(50)
    load_text.write("Grouping representative sentences...")
    if len(rep_sentences) <= 2:
        cluster_articles = [
            {
                "cluster_id": idx,
                "sentences": [rep_sentences[idx]],
                "representative": rep_sentences[idx],
                "representative_with_context": rep_sentences[idx]
            }
            for idx in range(len(rep_sentences))
        ]
    else:
        cluster_articles = group_representative_sentences(rep_sentences)
    del rep_sentences
    gc.collect()
    progress_bar.progress(55)
    load_text.write("Determining cluster validity and summarizing...")
    valid_clusters = []
    for i, cluster in enumerate(cluster_articles):
        if is_cluster_valid(cluster, keywords=keywords_bert, debug=True):
            cluster = objectify_and_summarize(cluster, light=False)
            valid_clusters.append(cluster)
    del cluster_articles
    gc.collect()
    progress_bar.progress(75)
    load_text.write("Calculating reliability of valid clusters...")
    valid_clusters = calculate_reliability(valid_clusters)
    progress_bar.progress(85)
    load_text.write("Determining cluster keywords...")
    visual_keywords = []
    for cluster in valid_clusters:
        kw_model = KeyBERT()
        keywords_bert = kw_model.extract_keywords(
            cluster["summary"], keyphrase_ngram_range=(1, 3), stop_words="english", top_n=15
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
    all_dates = []
    for cluster in valid_clusters:
        for sentence in cluster["sentences"]:
            d_str = getattr(sentence, "date", None)
            if d_str:
                try:
                    d_parsed = parser.parse(str(d_str))
                    all_dates.append(d_parsed)
                except:
                    pass
    if len(all_dates) > 0:
        global_min_date = min(all_dates)
        global_max_date = max(all_dates)
    else:
        global_min_date = None
        global_max_date = None
    with st.container():
        st.write(f"#### Analysis Results about {keywords[0].title()}...")
        for i, cluster in enumerate(valid_clusters):
            with st.expander(f"Cluster regarding **{unique_cluster_keywords[i].title()}**...", i == 0):
                st.write("##### Situation Rundown:")
                st.write(cluster["summary"])
                stoggle_text = "<ul>"
                for sentence in cluster["sentences"]:
                    stoggle_text += f"<li><b>[Source: {sentence.source}] [Date: {sentence.date}]:</b> {sentence.text}</li>"
                stoggle_text += "</ul>"
                stoggle("All Sentences (Click to Expand)", stoggle_text.strip())
                st.write("----")
                reliability = cluster["reliability"]
                meter_value = min(reliability, 50)
                if reliability <= 5:
                    reliability_text = "Very Reliable"
                    reliability_color = "green"
                elif reliability <= 15:
                    reliability_text = "Reliable"
                    reliability_color = "limegreen"
                elif reliability <= 25:
                    reliability_text = "Somewhat Reliable"
                    reliability_color = "orange"
                elif reliability <= 35:
                    reliability_text = "Somewhat Unreliable"
                    reliability_color = "orangered"
                else:
                    reliability_text = "Unreliable"
                    reliability_color = "darkred"
                st.markdown(f"**Reliability:** <span style='color:{reliability_color}'>{reliability_text} (Score: {round(reliability, 3)})</span>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div style='width: 100%; height: 20px; background: linear-gradient(to right,
                        green 0%, green 10%,
                        limegreen 10%, limegreen 30%,
                        orange 30%, orange 50%,
                        orangered 50%, orangered 70%,
                        darkred 70%, darkred 100%);
                    border-radius: 10px; position: relative; margin-top: 10px;'>
                        <div style='position: absolute; top: 0; left: {(meter_value / 50) * 100}%; 
                            width: 2px; height: 100%; background-color: black;'></div>
                    </div><br>
                    """, unsafe_allow_html=True)
                stats = cluster.get("reliability_stats", {})
                old_obj = stats.get("objectivity", 0)
                new_obj = 1 - 2 * (old_obj - 0.75)
                if new_obj < 0:
                    new_obj = 0
                if new_obj > 1:
                    new_obj = 1
                obj_val = round(new_obj, 3)
                sources = sorted(list(set([s.source for s in cluster["sentences"] if hasattr(s, "source")])))
                date_strings = [s.date for s in cluster["sentences"] if hasattr(s, "date")]
                parsed_dates = []
                for d_str in date_strings:
                    try:
                        parsed_dates.append(parser.parse(str(d_str)))
                    except:
                        pass
                if not global_min_date or not global_max_date:
                    timeline_div = "No valid dates found."
                else:
                    if len(parsed_dates) == 0:
                        timeline_div = "No valid dates found."
                    else:
                        min_date = global_min_date
                        max_date = global_max_date
                        total_days = (max_date - min_date).days
                        if total_days < 1:
                            total_days = 1
                        timeline_div = "<div style='width: 100%; height: 90px; position: relative; margin-top: 20px;'><div style='position: absolute; top: 50%; transform: translateY(-50%); width: 100%; height: 4px; background-color: #e0e0e0; border-radius: 2px;'></div>"
                        buffer = 5
                        offsets = []
                        for d in parsed_dates:
                            diff_days = (d - min_date).days
                            raw_offset = diff_days / total_days
                            scaled_offset = buffer + raw_offset * (100 - 2 * buffer)
                            label = d.strftime('%b %d, %Y')
                            offsets.append((scaled_offset, label))
                        offsets.sort(key=lambda x: x[0])
                        first_label = min_date.strftime('%b %d, %Y')
                        last_label = max_date.strftime('%b %d, %Y')
                        timeline_div += f"<div style='position: absolute; left: 0; top: 60%; font-size: 12px; white-space: nowrap;'>{first_label}</div><div style='position: absolute; right: 0; top: 60%; font-size: 12px; text-align: right; white-space: nowrap;'>{last_label}</div>"
                        for off, lbl in offsets:
                            timeline_div += f"<div style='position: absolute; left: {off}%; top: 50%; transform: translate(-50%, -50%);'><div style='width: 10px; height: 10px; background-color: black; border-radius: 50%;'></div><div style='position: absolute; top: -35px; left: 50%; transform: translateX(-50%); font-size: 12px; white-space: nowrap;'>{lbl}</div></div>"
                        if len(offsets) == 2:
                            off1, _ = offsets[0]
                            off2, _ = offsets[1]
                            if off1 != off2:
                                left_off = min(off1, off2)
                                width = abs(off2 - off1)
                                timeline_div += f"<div style='position: absolute; top: 50%; transform: translateY(-50%); left: {left_off}%; width: {width}%; height: 2px; border-bottom: 2px dotted black;'></div>"
                        if len(offsets) > 2:
                            left_off = offsets[0][0]
                            right_off = offsets[-1][0]
                            width = abs(right_off - left_off)
                            if width > 0:
                                timeline_div += f"<div style='position: absolute; top: 50%; transform: translateY(-50%); left: {left_off}%; width: {width}%; height: 2px; border-bottom: 2px dotted black;'></div>"
                        timeline_div += "</div>"
                src_list_html = ""
                for s in sources:
                    src_list_html += f"{s.title()}, "
                src_list_html = src_list_html.strip(", ")
                stoggle("Reliability Stats (Click to Expand)", f"<b>Objectivity Score:</b> {round(obj_val*100, 2)}/100<br><b>Sources:</b> {src_list_html}<br><b>Timeline:</b><br>{timeline_div}")
    progress_bar.empty()
    if "push_notification" not in st.session_state or st.session_state["push_notifications"]:
        make_notification("Article Analysis", "The analysis of the article is complete!")
        notif_text.empty()