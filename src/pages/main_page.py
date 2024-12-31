import streamlit as st
from streamlit_extras import colored_header

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from streamlit_extras.colored_header import colored_header

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
    color_name=st.session_state["header_color"] if "header_color" in st.session_state else "blue-70"
)

st.image("data/images/misinformation stats.png")

st.markdown("In recent years, misinformation has been on the rise. It comes in tandem with new technological innovations and growing polarization, and the dissemination of false, innacurate, and exaggerated informations has outstripped the ability to discern fact from fiction.")
st.markdown("Echo chambers, greater access to news, algorithms, and much more all serve as catalysts to accelerate the spread of misinformation. According to a survey done by Citizen Data in 2024, at least 83% of the surveyed individuals have reported seeing misinformation.")
st.markdown("The issues lies with the fact that the survey only considers what the reporting individuals recognize as misinformation, not all misinformation, meaning the numbers are likely much higher, both for the number of people seeing misinformation and the amount of misinformation people see.")
st.markdown("The consequences of misinformation can be devastating. Studies by the World Health Organization (WHO) found that misinformation can cause people to 'feel mental, social, political and/or economic distress' and also exacerbate issues such as polarization.")
st.markdown("It is evident that something needs to be done about misinformation, however small, which is why I launched the project Objective News.")
