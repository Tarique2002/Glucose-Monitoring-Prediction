import streamlit as st

def render_kpi_card(title: str, value: str, unit: str):
    """
    Renders a KPI card using native Streamlit containers.
    The CSS classes target these containers to apply glassmorphism.
    """
    with st.container():
        st.markdown(f"""
        <div class="kpi-container animate-fade-in">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)

def render_ai_assistant_button():
    """
    Renders the floating AI assistant button.
    """
    st.markdown("""
    <div class="ai-assistant-btn" title="AI Assistant">
        ✨
    </div>
    """, unsafe_allow_html=True)
