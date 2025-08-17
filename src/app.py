import streamlit as st
import os
import sys
from utility import ensure_nltk_data
if 'nltk' in sys.modules and not hasattr(sys.modules.get('nltk'), 'data'):
    del sys.modules['nltk']
ensure_nltk_data()

st.set_page_config(page_title="Objective News", page_icon="📰", layout="wide", initial_sidebar_state="expanded")

pages = [
    st.Page("pages/intro_page.py", title="Introduction"),
    st.Page("pages/article_analysis_page.py", title="Article Analysis"),
    st.Page("pages/objectivity_page.py", title="Objectivity Playground"),
    st.Page("pages/misc_page.py", title="Utilities Explorer"),
]

with st.sidebar:
    st.title("Objective News")
    st.write("Analyze topics across sources, cluster narratives, and see objective summaries with reliability scores.")
    pg = st.navigation(pages)

pg.run()
