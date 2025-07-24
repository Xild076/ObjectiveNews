import streamlit as st
import time
import os

st.set_page_config(
    page_title="News Analysis Engine",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/intro_page.py", title="Introduction"),
    st.Page("pages/article_analysis_page.py", title="Article Analysis"),
    st.Page("pages/objectivity_page.py", title="Objectivity Playground"),
    st.Page("pages/misc_page.py", title="Utilities Explorer"),
]

with st.sidebar:
    st.title("News Analysis Engine")
    st.write("An automated suite of tools to deconstruct news, analyze bias, and synthesize information.")
    pg = st.navigation(pages)

pg.run()
