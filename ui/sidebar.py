import streamlit as st
from streamlit_option_menu import option_menu

def render_sidebar() -> str:
    """
    Renders the modern sidebar navigation using streamlit-option-menu.
    Returns the selected menu option.
    """
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <div style='background: linear-gradient(135deg, #2563EB, #14B8A6); padding: 3px; border-radius: 50%; width: 86px; margin: 0 auto 10px auto;'>
                    <img src='https://api.dicebear.com/7.x/avataaars/svg?seed=Health&backgroundColor=ffffff' style='border-radius: 50%; width: 80px; height: 80px;'>
                </div>
                <h3 style='margin:0; font-size: 1.2rem; color: #0F172A;'>AI Health Pro</h3>
                <p style='margin:0; font-size: 0.9rem; color: #64748B;'>Clinician Portal</p>
            </div>
        """, unsafe_allow_html=True)
        
        menu = option_menu(
            menu_title=None,
            options=[
                "Home & Predict", 
                "Lifestyle Simulator",
                "Explainability (SHAP)",
                "Analytics Dashboard", 
                "Model Performance",
                "Prediction History", 
                "AI Assistant"
            ],
            icons=[
                "house", 
                "sliders",
                "magic",
                "bar-chart-line", 
                "speedometer2",
                "clock-history", 
                "robot"
            ],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "var(--primary)", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px", 
                    "text-align": "left", 
                    "margin":"0px", 
                    "--hover-color": "rgba(37, 99, 235, 0.1)"
                },
                "nav-link-selected": {"background-color": "rgba(37, 99, 235, 0.15)", "color": "var(--primary)", "font-weight": "600"},
            }
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("✅ **System Status:** All operational")
        
    return menu
