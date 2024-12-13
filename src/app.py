import streamlit as st
from streamlit_extras.colored_header import colored_header

main_page = st.Page("pages/main_page.py", title="Main Page", default=True)
documentation_page = st.Page("pages/documentation_page.py", title="Documentation Page")

article_page = st.Page("pages/article_page.py", title="Article Analysis Page")
objectify_page = st.Page("pages/objectify_page.py", title="Objectify Page")

pg = st.navigation({
    "Overview": [main_page, documentation_page],
    "Tools": [article_page, objectify_page]
})

pg.run()