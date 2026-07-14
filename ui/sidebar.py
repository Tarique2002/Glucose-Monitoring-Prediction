import streamlit as st

def render_sidebar() -> str:
    """
    Renders the sidebar navigation and profile.
    Returns the selected menu option.
    """
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <div style='background: linear-gradient(135deg, #2563EB, #14B8A6); padding: 3px; border-radius: 50%; width: 86px; margin: 0 auto 10px auto;'>
                    <img src='https://api.dicebear.com/7.x/avataaars/svg?seed=Felix&backgroundColor=ffffff' style='border-radius: 50%; width: 80px; height: 80px;'>
                </div>
                <h3 style='margin:0; font-size: 1.2rem; color: #0F172A;'>Dr. Alex Chen</h3>
                <p style='margin:0; font-size: 0.9rem; color: #64748B;'>Endocrinologist</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Navigation")
        menu = st.radio(
            "", 
            ["🏠 Home & Predict", "📊 Dashboard Analytics", "🧑‍⚕️ Patient History", "🤖 AI Health Insights"], 
            label_visibility="collapsed"
        )
        
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='background: rgba(37,99,235,0.1); padding: 15px; border-radius: 12px; border: 1px solid rgba(37,99,235,0.2);'>
                <p style='margin:0; font-size:0.8rem; color: #3B82F6; font-weight: 600;'>SYSTEM STATUS</p>
                <p style='margin:0; font-size:0.9rem; color: #0F172A;'>✅ All systems operational</p>
                <p style='margin:0; font-size:0.75rem; color: #64748B;'>Last updated: Just now</p>
            </div>
        """, unsafe_allow_html=True)
        
    return menu
