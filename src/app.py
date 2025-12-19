import streamlit as st
import os
import sys
from utility import ensure_nltk_data
if 'nltk' in sys.modules and not hasattr(sys.modules.get('nltk'), 'data'):
    del sys.modules['nltk']
ensure_nltk_data()

st.set_page_config(page_title="Objective News", page_icon="📰", layout="wide", initial_sidebar_state="expanded")

# Global light styling
st.markdown(
    """
    <style>
    body {background: radial-gradient(circle at 20% 20%, rgba(59,130,246,0.08), transparent 25%),
                   radial-gradient(circle at 80% 0%, rgba(16,185,129,0.08), transparent 20%);
          }
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .stButton>button {border-radius: 10px; font-weight: 600;}
    .metric-card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; background: #f8fafc;}
    </style>
    """,
    unsafe_allow_html=True,
)

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
