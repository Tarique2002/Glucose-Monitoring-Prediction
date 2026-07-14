import streamlit as st
from ui.styles import inject_custom_css
from ui.sidebar import render_sidebar
from ui.prediction_page import render_prediction_page
from ui.dashboard import render_dashboard
from ui.charts import create_feature_importance_chart
from ui.components import render_ai_assistant_button
from src.prediction import load_and_train

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Glucore | Premium AI Healthcare",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS INJECTION ---
inject_custom_css()

# --- MODEL LOADING & TRAINING ---
# Loads model once and caches it using @st.cache_data in src.prediction
data, reg_model, clf_model, scaler_reg, scaler_clf, reg_mae, reg_r2, clf_acc, feature_imp, reg_features = load_and_train()

# --- SIDEBAR NAVIGATION ---
menu = render_sidebar()

# --- MAIN APP ROUTING ---
if menu == "🏠 Home & Predict":
    render_prediction_page(reg_model, clf_model, scaler_reg, scaler_clf)

elif menu == "📊 Dashboard Analytics":
    render_dashboard(data, clf_acc)

elif menu == "🧑‍⚕️ Patient History":
    st.markdown("<h2 class='animate-fade-in'>Patient Records Database</h2>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card animate-fade-in' style='animation-delay: 0.1s;'>", unsafe_allow_html=True)
    st.markdown("Easily search and filter through the historical clinical dataset used for training the model.")
    
    st.dataframe(
        data.style.background_gradient(cmap='Blues', subset=['Glucose'])
                  .background_gradient(cmap='Reds', subset=['BMI']),
        use_container_width=True,
        height=600
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "🤖 AI Health Insights":
    st.markdown("<h2 class='animate-fade-in'>AI Diagnostic Insights</h2>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 24px;'>
        <div class="glass-card animate-fade-in" style='animation-delay: 0.1s;'>
            <div style="display:flex; align-items:center; gap: 15px;">
                <div style="font-size: 40px; background: rgba(37,99,235,0.1); padding: 10px; border-radius: 20px;">🧠</div>
                <div>
                    <h3 style="margin:0; font-size:1.2rem;">Feature Importance</h3>
                    <p style="margin:0; color: var(--text-muted); font-size: 0.9rem;">What drives predictions</p>
                </div>
            </div>
            <p style="margin-top: 15px; font-size: 0.95rem; line-height: 1.5; color: #334155;">
                The model relies heavily on <b>Glucose</b> levels, followed by <b>BMI</b> and <b>Age</b>. Skin thickness has the least predictive power for diabetes onset in this specific demographic.
            </p>
        </div>
        
        <div class="glass-card animate-fade-in" style='animation-delay: 0.2s;'>
            <div style="display:flex; align-items:center; gap: 15px;">
                <div style="font-size: 40px; background: rgba(34,197,94,0.1); padding: 10px; border-radius: 20px;">⚕️</div>
                <div>
                    <h3 style="margin:0; font-size:1.2rem;">Model Reliability</h3>
                    <p style="margin:0; color: var(--text-muted); font-size: 0.9rem;">Clinical validation metrics</p>
                </div>
            </div>
            <p style="margin-top: 15px; font-size: 0.95rem; line-height: 1.5; color: #334155;">
                Classification accuracy stands at <b>{clf_acc*100:.1f}%</b>. The regression model predicts glucose with an average error of <b>{reg_mae:.1f} mg/dL</b>. Recommended for supplementary clinical screening only.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card animate-fade-in' style='animation-delay: 0.3s;'>", unsafe_allow_html=True)
    st.markdown("<h4>AI Feature Drivers (Random Forest)</h4>", unsafe_allow_html=True)
    fig_imp = create_feature_importance_chart(feature_imp)
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- FLOATING AI ASSISTANT ---
render_ai_assistant_button()
