import streamlit as st
from streamlit_extras.colored_header import colored_header

main_page = st.Page("pages/main_page.py", title="Main Page", default=True)
documentation_page = st.Page("pages/documentation_page.py", title="Documentation")

article_page = st.Page("pages/article_page.py", title="Article Analysis")
objectify_page = st.Page("pages/objectify_page.py", title="Objectify Text")
grouping_page = st.Page("pages/grouping_page.py", title="Grouping Text")
misc_page = st.Page("pages/misc_page.py", title="Misc Tools")

pg = st.navigation({
    "Overview": [main_page, documentation_page],
    "Power Tools": [article_page, objectify_page],
    "Mini Tools": [grouping_page, misc_page]
})

pg.run()