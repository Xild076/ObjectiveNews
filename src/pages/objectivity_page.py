import streamlit as st
from objectify.objectify import objectify_text, calculate_objectivity
import difflib

st.set_page_config(page_title="Objectivity Playground", layout="wide")

def render_diff_html(original, new):
    sm = difflib.SequenceMatcher(None, original.split(), new.split())
    output_html = ""
    for opcode, i1, i2, j1, j2 in sm.get_opcodes():
        if opcode == 'equal':
            output_html += " " + " ".join(original.split()[i1:i2])
        elif opcode == 'replace':
            output_html += f" <span style='background-color:#ffebe9;color:#d9534f;text-decoration:line-through;padding:2px 4px;border-radius:4px;'>{' '.join(original.split()[i1:i2])}</span>"
            output_html += f" <span style='background-color:#dbeafe;color:#2563eb;font-weight:bold;padding:2px 4px;border-radius:4px;'>{' '.join(new.split()[j1:j2])}</span>"
        elif opcode == 'delete':
            output_html += f" <span style='background-color:#ffebe9;color:#d9534f;text-decoration:line-through;padding:2px 4px;border-radius:4px;'>{' '.join(original.split()[i1:i2])}</span>"
        elif opcode == 'insert':
            output_html += f" <span style='background-color:#dbeafe;color:#2563eb;font-weight:bold;padding:2px 4px;border-radius:4px;'>{' '.join(new.split()[j1:j2])}</span>"
    return output_html.strip()

st.title("Objectivity Playground")
st.markdown("Test our objectification engine. Enter a subjective sentence and see how the system rewrites it to be more neutral.")
st.markdown("""
**How to read the results:**
- **Objectivity Score:** A score from 0.0 to 1.0, where 1.0 is completely neutral.
- <span style='color:#d9534f;text-decoration:line-through;'>Red text</span> was removed for being too subjective.
- <span style='background-color:#dbeafe;color:#2563eb;font-weight:bold;padding:2px 4px;border-radius:4px;'>Blue text</span> was added or modified to be more neutral.
""", unsafe_allow_html=True)
st.divider()

default_sentence = "The corrupt politician delivered a disastrous speech, infuriating the brave citizens who are fighting for justice."
text_input = st.text_area("**Enter a sentence to objectify**", value=default_sentence, height=100, help="Paste a subjective sentence from any source.")

import gc
if st.button("Objectify Sentence", use_container_width=True, type="primary"):
    if text_input:
        with st.spinner("Analyzing and rewriting..."):
            original_score = calculate_objectivity(text_input)
            objectified_sentence = objectify_text(text_input)
            new_score = calculate_objectivity(objectified_sentence)
            diff_html = render_diff_html(text_input, objectified_sentence)
        st.divider()
        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Original Objectivity Score", value=f"{original_score:.3f}")
        with col2:
            st.metric(label="New Objectivity Score", value=f"{new_score:.3f}", delta=f"{new_score - original_score:.3f}")
        with st.container(border=True):
            st.markdown("**Detailed Transformation View**")
            st.markdown(f'<div style="padding:10px;line-height:1.7;">{diff_html}</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Final Neutral Text**")
            st.write(objectified_sentence)
        st.success("Objectification complete! Review the changes above.")
        for k in tuple(locals().keys()):
            if k not in ('st', 'gc'):
                del locals()[k]
        gc.collect()
    else:
        st.warning("Please enter a sentence.")