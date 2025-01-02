import streamlit as st
from streamlit_extras import colored_header

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tab1, tab2, tab3 = st.tabs(["README", "Paper", "License"])

with tab1:
    with open("README.md", "r") as f:
        st.markdown(f.read())

with tab2:
    with open("paper/paper_v1-0.md", "r") as f:
        st.markdown(f.read())

with tab3:
    with open("LICENSE", "r") as f:
        st.markdown(f.read())