import streamlit as st
from streamlit_extras import colored_header

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.title("Introduction to Alitheia AI")

st.markdown("""
    <h4 style='font-weight: normal; color: #5D6D7E; text-align: center;'>
        Aiming towards finding and presenting the most objective news, texts, and information possible in a growing world of misinformation.
    </h4>
    """, unsafe_allow_html=True)

colored_header.colored_header(
    label="The Status Quo of Misinformation",
    description="A quick description of the current state of misinformation.",
    color_name=st.session_state["header_color"] if "header_color" in st.session_state else "blue-70"
)

st.image(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/images/misinformation_stats.png')))

st.markdown("In recent years, misinformation has been on the rise. It comes in tandem with new technological innovations and growing polarization, and the dissemination of false, innacurate, and exaggerated informations has outstripped the ability to discern fact from fiction.")
st.markdown("Echo chambers, greater access to news, algorithms, and much more all serve as catalysts to accelerate the spread of misinformation. According to a survey done by Citizen Data in 2024, at least 83% of the surveyed individuals have reported seeing misinformation.")
st.markdown("The issues lies with the fact that the survey only considers what the reporting individuals recognize as misinformation, not all misinformation, meaning the numbers are likely much higher, both for the number of people seeing misinformation and the amount of misinformation people see.")
st.markdown("The consequences of misinformation can be devastating. Studies by the World Health Organization (WHO) found that misinformation can cause people to 'feel mental, social, political and/or economic distress' and also exacerbate issues such as polarization.")
st.markdown("It is evident that something needs to be done about misinformation, however small, which is why I launched the project Alitheia AI.")

colored_header.colored_header(
    label="What is Alitheia AI?",
    description="A brief overview of the project and its goals.",
    color_name=st.session_state["header_color"] if "header_color" in st.session_state else "blue-70"
)

st.markdown("""
Alitheia AI is a sophisticated pipeline designed to combat misinformation by deconstructing news on any given topic. It provides an automated, objective-first approach to understanding the world by synthesizing multiple sources into clear, unbiased, and reliable narratives.
""")

st.subheader("How It Works")
st.markdown("""
When you provide a topic or article, the engine executes a multi-step analysis to deliver insightful results. This process is designed to move from a wide range of chaotic information to a set of clear, structured, and trustworthy narratives.
""")

st.markdown("""
1.  **Gathers Intelligence**: Instead of you needing to read a dozen different articles, the engine does it for you. It automatically fetches a diverse set of news articles related to your topic, intentionally avoiding ideological echo chambers.

2.  **Identifies Core Narratives**: Using a custom-trained grouping model, the engine reads through all the collected texts to identify the main underlying themes or "narratives." This step distills the essential points from all the noise.

3.  **Rewrites for Objectivity**: Each narrative is analyzed for biased, subjective, and emotionally-charged language. These words are then replaced with neutral, objective synonyms, stripping away the spin and leaving the factual core of the story.

4.  **Assigns a Reliability Score**: Finally, each neutralized narrative is given a comprehensive reliability score. This isn't a simple "true/false" rating, but a nuanced metric calculated from:
    *   The **reputation** of the contributing news sources.
    *   The **objectivity** of the language used.
    *   The **recency** and **coverage** of the reporting.
""")

st.subheader("Our Solution to Key Problems")
st.markdown("""
This entire process is engineered to directly address the primary challenges of modern news consumption:
""")
st.markdown("""
-   **Information Overload**: We synthesize vast amounts of information into a few key, digestible narratives, saving you time and effort.
-   **Hidden Bias**: Our objectification engine actively identifies and neutralizes biased language, presenting you with the facts, not the spin.
-   **Uncertain Credibility**: We provide a transparent, multi-factor reliability score to help you gauge the trustworthiness of the information you're reading.
""")