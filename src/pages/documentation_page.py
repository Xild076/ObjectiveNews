import streamlit as st
from streamlit_extras import colored_header

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tab1, tab2 = st.tabs(["README", "Paper"])

with tab1:
    with open("README.md", "r") as f:
        st.markdown(f.read())

with tab2:
    st.markdown("Paper is currently being updated!")