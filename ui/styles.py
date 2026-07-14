import streamlit as st
import os

def inject_custom_css():
    """
    Reads the main CSS file and injects it into the Streamlit app.
    """
    css_path = os.path.join(os.path.dirname(__file__), '..', 'css', 'main.css')
    try:
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Running with default styles.")
