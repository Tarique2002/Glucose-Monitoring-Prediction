import streamlit as st

def render_sidebar() -> str:
    """
    Renders the sidebar navigation and profile.
    Returns the selected menu option.
    """
    with st.sidebar:
        # User Profile using Streamlit Native components as much as possible
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <div style='background: linear-gradient(135deg, #2563EB, #14B8A6); padding: 3px; border-radius: 50%; width: 86px; margin: 0 auto 10px auto;'>
                    <img src='https://api.dicebear.com/7.x/avataaars/svg?seed=Health&backgroundColor=ffffff' style='border-radius: 50%; width: 80px; height: 80px;'>
                </div>
                <h3 style='margin:0; font-size: 1.2rem; color: #0F172A;'>Clinical Dashboard</h3>
                <p style='margin:0; font-size: 0.9rem; color: #64748B;'>Authorized User</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Navigation")
        menu = st.radio(
            "", 
            ["🏠 Home & Predict", "📊 Dashboard Analytics", "🧑‍⚕️ Patient History", "🤖 AI Health Insights"], 
            label_visibility="collapsed"
        )
        
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        st.info("✅ **System Status:** All operational")
        
    return menu
