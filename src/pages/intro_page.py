import streamlit as st
from streamlit_extras import colored_header

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.title("Objective News")
st.caption("Multi-source clustering · Bias neutralization · Reliability scoring")

st.markdown(
    """
    <style>
    .hero {
        padding: 18px 22px;
        border-radius: 14px;
        background: linear-gradient(120deg, rgba(59,130,246,0.12), rgba(16,185,129,0.12));
        border: 1px solid #e5e7eb;
    }
    .pill {display:inline-block;padding:4px 10px;border-radius:999px;background:#e0e7ff;color:#1e3a8a;font-weight:600;font-size:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

hero = st.container()
with hero:
    st.markdown("<div class='pill'>Objective-first pipeline</div>", unsafe_allow_html=True)
    h1, h2 = st.columns([2.5, 1.5])
    with h1:
        st.subheader("From messy headlines to objective narratives")
        st.write("We fetch diverse sources, cluster narratives, neutralize bias, and score reliability so you can trust what you read.")
        cta1, cta2, cta3 = st.columns([1.2,1.2,1.2])
        if cta1.button("Analyze Articles", use_container_width=True, type="primary"):
            st.switch_page("pages/article_analysis_page.py")
        if cta2.button("Objectivity Lab", use_container_width=True):
            st.switch_page("pages/objectivity_page.py")
        if cta3.button("Utilities", use_container_width=True):
            st.switch_page("pages/misc_page.py")
    with h2:
        st.image(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/images/misinformation_stats.png')))

st.markdown("---")

st.subheader("Why this matters")
col1, col2 = st.columns(2)
with col1:
    colored_header.colored_header(
        label="Misinformation is rising",
        description="83% of people report seeing misinformation (Citizen Data, 2024).",
        color_name="blue-70",
    )
    st.write("Algorithms, echo chambers, and volume make it hard to separate fact from spin.")
with col2:
    colored_header.colored_header(
        label="We counter with objectivity",
        description="Fetch diverse sources, cluster, neutralize language, and score reliability.",
        color_name="green-70",
    )
    st.write("The pipeline reduces noise, exposes bias, and surfaces trustworthy narratives.")

st.markdown("---")

st.subheader("How it works")
steps = [
    ("Gather & filter", "Fetch diverse articles, avoid echo chambers, and keep relevant, dense sentences."),
    ("Cluster & summarize", "Group narratives, merge overlaps, and summarize cleanly."),
    ("Neutralize & score", "Rewrite to objective language and score reliability across domains."),
]
c1, c2, c3 = st.columns(3)
for (title, desc), col in zip(steps, [c1, c2, c3]):
    with col.container(border=True):
        st.markdown(f"**{title}**")
        st.write(desc)

st.markdown("---")

st.subheader("Quick start")
qs1, qs2, qs3 = st.columns(3)
with qs1:
    st.button("Analyze a topic", use_container_width=True, type="primary", on_click=lambda: st.switch_page("pages/article_analysis_page.py"))
with qs2:
    st.button("Try objectivity lab", use_container_width=True, on_click=lambda: st.switch_page("pages/objectivity_page.py"))
with qs3:
    st.button("Open utilities", use_container_width=True, on_click=lambda: st.switch_page("pages/misc_page.py"))