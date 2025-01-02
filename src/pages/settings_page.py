import streamlit as st
from streamlit_extras import colored_header
from streamlit_extras.button_selector import button_selector

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

colored_header.colored_header(
    "Settings",
    "Change your settings!",
    st.session_state["header_color"] if "header_color" in st.session_state else "blue-70"
)

st.write("### Color Accents")

color_option = button_selector(
    label="Header Accent Color",
    options=["Blue", "Blue-Green", "Green", "Light Blue", "Orange", "Red", "Violet", "Yellow"],
    index=0
)
COLOR_INDEX = ["blue-70", "blue-green-70", "green-70", "light-blue-70", "orange-70", "red-70", "violet-70", "yellow-70"]

update_button = st.button("Update Settings")

if update_button:
    st.session_state["header_color"] = COLOR_INDEX[color_option]

st.write("### Notifications")

def checkbox_on_change():
    st.session_state["push_notifications"] = notif_checkbox

notif_checkbox = st.checkbox("Enable Push Notifications", st.session_state["push_notifications"] if "push_notifications" in st.session_state else True, on_change=checkbox_on_change)