import streamlit as st
from streamlit_extras.colored_header import colored_header

st.set_page_config(
    page_title="Objective News",
    layout="centered",
)

st.markdown("""
    <h1 style='font-weight: normal; color: #2E86C1; text-align: center;'>
        Objective News
    </h1>
    """, unsafe_allow_html=True)

st.markdown("""
    <h4 style='font-weight: normal; color: #5D6D7E; text-align: center;'>
        Aiming towards finding and creating the most objective news, texts, and information possible in a growing world of misinformation.
    </h4>
    """, unsafe_allow_html=True)

colored_header(
    label="The Status Quo of Misinformation",
    description="A quick description of the current state of misinformation.",
    color_name="light-blue-70"
)

st.image("images/misinformation stats.png")

st.markdown("In recent years, misinformation has been on the rise. It comes in tandem with new technological innovations and growing polarization, and the dissemination of false, innacurate, and exaggerated informations has outstripped the ability to discern fact from fiction.")
st.markdown("Echo chambers, greater access to news, algorithms, and much more all serve as catalysts to accelerate the spread of misinformation. According to a survey done by Citizen Data in 2024, at least 83% of the surveyed individuals have reported seeing misinformation.")
st.markdown("The issues lies with the fact that the survey only considers what the reporting individuals recognize as misinformation, not all misinformation, meaning the numbers are likely much higher, both for the number of people seeing misinformation and the amount of misinformation people see.")
st.markdown("The consequences of misinformation can be devastating. Studies by the World Health Organization (WHO) found that misinformation can cause people to 'feel mental, social, political and/or economic distress' and also exacerbate issues such as polarization.")
st.markdown("It is evident that something needs to be done about misinformation, however small, which is why I launched the project Objective News.")

colored_header(
    label="Tools Provided by Objective News",
    description="A quick overview of what tools are provided by objective news to help combat misinformation.",
    color_name="light-blue-70"
)

tabs = st.tabs(["Article Analysis", "Objectify Text", "Grouping Text", "Misc Tools"])

with tabs[0]:
    with st.container(border=1):
        st.subheader("Power Tool - Article Analysis")
        st.markdown("**This tool allows you to input a link and retrieve information about the topic discussed in the article.**")
        st.image("images/Article.png")
        st.markdown("The tool will organize all the information found across the articles located into groups, summarize and objectify the information, and detail the reliability of such information. *The loading time of this tool is about 1-2 minutes per topic. Optimizations are currenlty being made to improve this tool.*")

with tabs[1]:
    with st.container(border=1):
        st.subheader("Power Tool - Objectify Text")
        st.markdown("**This tool allows you to input text and get its objective version.**")
        st.image("images/Objectify.png")
        st.markdown("The tool will remove strong words and, if necessary, replace them with their most objective synonyms in an attempt to create the most objective text. Currently, this is the least consistent tool with many issues due to grammar rules. Please leave feedback through the feedback function if there are any issues. *The loading time of this tool is about 15-45 seconds per text. Optimizations are currently being made.*")

with tabs[2]:
    with st.container(border=1):
        st.subheader("Mini Tool - Grouping Text")
        st.markdown("**This tool allows you to group text into main ideas.**")
        st.image("images/Grouping.png")
        st.markdown("The tool will group text into main ideas based on general similarity of issues discussed. There are many configurations with the default ones provided. *The loading time of the tool is about 1-5 seconds per text.*")

with tabs[3]:
    with st.container(border=1):
        st.subheader("Mini Tool - Misc")
        st.markdown("**There are two tools in this tab: The synonym finder tool and the summarizer tool.**")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=1):
                st.markdown("#### Synonym Finder")
                st.markdown("This tool will provide you the synonyms of the given word organized based on their objectivity, their part of speech, or their text length. *The loading time of the tool is about 1-5 seconds per text.*")
        with col2:
            with st.container(border=1):
                st.markdown("#### Summarizer")
                st.markdown("This tool will summarize given text using the facebook/bart-large model found on huggingface. There are three options for summarization: short, medium, and long summarizations. *The loading time of the tool is about 10-15 seconds per text.*")