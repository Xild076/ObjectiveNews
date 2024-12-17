import streamlit as st

tab1, tab2 = st.tabs(["README", "Paper"])

with tab1:
    with open("README.md", "r") as f:
        st.markdown(f.read())

with tab2:
    with open("paper.md", "r") as f:
        st.markdown(f.read())