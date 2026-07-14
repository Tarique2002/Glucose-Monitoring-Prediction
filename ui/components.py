import streamlit as st

def render_kpi_card(title: str, value: str, unit: str):
    """
    Renders a glassmorphism KPI card.
    """
    st.markdown(f"""
    <div class="glass-card animate-fade-in">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">{unit}</div>
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
