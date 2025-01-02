import streamlit as st

st.set_page_config(page_title="Objective News", page_icon="data/images/objectivenews.ico")

import streamlit.components.v1 as components

components.html("""
<head>
    <title>Objective New</title>
    <meta name="description" content="Aiming towards finding and creating the most objective news, texts, and information possible in a growing world of misinformation.">
    <meta name="keywords" content="Streamlit, Objective New, Summarize, Objectify, Group, Text, News, Article, Information, Misinformation, Disinformation, Fake News">
    <meta name="author" content="Xild076 (Harry Yin)">
</head>
""", height=0)

@st.cache_resource
def load_pages():
    main_page = st.Page("pages/main_page.py", title="Objective News - Main Page - By Harry Yin", default=True)
    documentation_page = st.Page("pages/documentation_page.py", title="Objective News - Documentation - By Harry Yin")
    
    article_page = st.Page("pages/article_page.py", title="Objective News - Article Analysis - By Harry Yin")
    objectify_page = st.Page("pages/objectify_page.py", title="Objective News - Objectify Text - By Harry Yin")
    grouping_page = st.Page("pages/grouping_page.py", title="Objective News - Grouping Text - By Harry Yin")
    misc_page = st.Page("pages/misc_page.py", title="Objective News - Misc Tools - By Harry Yin")
    settings_page = st.Page("pages/settings_page.py", title="Objective News - Settings")
    
    return {
        "Overview": [main_page, documentation_page],
        "Power Tools": [article_page, objectify_page],
        "Mini Tools": [grouping_page, misc_page],
        "Settings": [settings_page]
    }

pg = st.navigation(load_pages())
pg.run()