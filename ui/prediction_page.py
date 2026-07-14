import streamlit as st
import time
from src.prediction import predict_patient
from ui.charts import create_gauge_chart
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def render_prediction_page(
    reg_model: RandomForestRegressor, 
    clf_model: RandomForestClassifier, 
    scaler_reg: StandardScaler, 
    scaler_clf: StandardScaler
):
    """
    Renders the home page with the prediction form and results.
    """
    # Hero Section
    st.markdown("""
    <div class="hero-container animate-fade-in">
        <div class="hero-text">
            <div style="display:inline-block; padding: 6px 12px; background: rgba(37,99,235,0.1); color: var(--primary); border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin-bottom: 1rem; border: 1px solid rgba(37,99,235,0.2);">
                ✨ AI-Powered Healthcare
            </div>
            <h1>Next-Gen Glucose<br>Monitoring</h1>
            <p>Empower your clinical decisions with high-precision machine learning. Real-time predictions, continuous risk assessment, and personalized insights.</p>
        </div>
        <div class="hero-img" style="font-size: 140px; filter: drop-shadow(0 20px 30px rgba(0,0,0,0.15));" class="floating-3d">
            <div class="floating-3d">🧬</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input Form Layout
    st.markdown("<h2 style='margin-top: 2rem; margin-bottom: 1.5rem;'>Patient Parameters</h2>", unsafe_allow_html=True)
    
    form_col1, form_col2 = st.columns([1, 1.5])
    
    with form_col1:
        st.markdown("<div class='glass-card animate-fade-in' style='animation-delay: 0.1s;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom:1rem; color: var(--primary);'>📋 Medical History</h4>", unsafe_allow_html=True)
        
        pregnancies = st.slider("Pregnancies", 0, 17, 1)
        age = st.slider("Age", 18, 90, 30)
        bmi = st.slider("BMI", 10.0, 70.0, 26.0, 0.1)
        
        st.markdown("<br><h4 style='margin-bottom:1rem; color: var(--accent);'>🔬 Vitals & Labs</h4>", unsafe_allow_html=True)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 20, 130, 72)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 29)
        insulin = st.slider("Insulin (mu U/ml)", 0, 850, 80)
        dpf = st.slider("Diabetes Pedigree Function", 0.05, 2.50, 0.47, 0.01)
        st.markdown("</div>", unsafe_allow_html=True)
        
        predict_btn = st.button("Generate AI Prediction")

    with form_col2:
        if predict_btn:
            with st.spinner('Analyzing physiological data with Random Forest Models...'):
                time.sleep(0.8) # Simulate processing for premium feel
                
                inputs = {
                    'pregnancies': pregnancies,
                    'blood_pressure': blood_pressure,
                    'skin_thickness': skin_thickness,
                    'insulin': insulin,
                    'bmi': bmi,
                    'dpf': dpf,
                    'age': age
                }
                
                results = predict_patient(inputs, reg_model, clf_model, scaler_reg, scaler_clf)
                
                predicted_glucose = results['predicted_glucose']
                diabetes_risk = results['diabetes_risk']
                risk_proba = results['risk_proba']
                
                is_high_risk = predicted_glucose >= 126 or diabetes_risk == 1
                card_class = "result-danger" if is_high_risk else "result-safe"
                risk_color = "var(--danger)" if is_high_risk else "var(--accent)"
                risk_text = "High Risk Detected" if is_high_risk else "Normal Range"
                
                confidence_score = (risk_proba[1] if is_high_risk else risk_proba[0])*100
                
                st.markdown(f"""
                <div class="result-card {card_class} animate-fade-in" style='animation-delay: 0.2s;'>
                    <div style="font-size: 80px; margin-bottom: -10px;" class="floating-3d">
                        {'🚨' if is_high_risk else '✅'}
                    </div>
                    <h2 style="font-size: 3rem; margin-top: 10px; color: {risk_color}">{predicted_glucose:.1f} <span style="font-size: 1rem; color: var(--text-muted);">mg/dL</span></h2>
                    <h4 style="color: {risk_color}; letter-spacing: 1px; text-transform: uppercase;">{risk_text}</h4>
                    <p style="color: var(--text-muted); margin-bottom: 2rem;">Confidence Score: {confidence_score:.1f}%</p>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 2rem; text-align: left;">
                        <div style="background: rgba(255,255,255,0.6); padding: 15px; border-radius: 15px;">
                            <div style="font-size: 24px;">🥗</div>
                            <div style="font-weight: 600; font-size: 0.9rem; margin-top: 5px;">Diet</div>
                            <div style="font-size: 0.75rem; color: var(--text-muted); line-height: 1.3;">{"Strict low GI foods" if is_high_risk else "Maintain balanced diet"}</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.6); padding: 15px; border-radius: 15px;">
                            <div style="font-size: 24px;">🏃</div>
                            <div style="font-weight: 600; font-size: 0.9rem; margin-top: 5px;">Activity</div>
                            <div style="font-size: 0.75rem; color: var(--text-muted); line-height: 1.3;">{"Daily cardio required" if is_high_risk else "Regular exercise"}</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.6); padding: 15px; border-radius: 15px;">
                            <div style="font-size: 24px;">💊</div>
                            <div style="font-weight: 600; font-size: 0.9rem; margin-top: 5px;">Action</div>
                            <div style="font-size: 0.75rem; color: var(--text-muted); line-height: 1.3;">{"Consult physician immediately" if is_high_risk else "Routine checkup"}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge Chart
                fig_gauge = create_gauge_chart(predicted_glucose, risk_color)
                
                st.markdown("<div class='glass-card animate-fade-in' style='animation-delay: 0.3s; margin-top: 24px;'>", unsafe_allow_html=True)
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            # Idle State for right column
            st.markdown("""
            <div style="height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 4rem; opacity: 0.6; background: rgba(255,255,255,0.4); border-radius: 30px; border: 2px dashed rgba(203, 213, 225, 0.8);">
                <div style="font-size: 80px; margin-bottom: 20px; filter: grayscale(100%);" class="floating-3d">🩺</div>
                <h3 style="color: var(--text-muted); font-weight: 500;">Awaiting Input</h3>
                <p style="text-align: center; color: var(--text-muted); font-size: 0.9rem;">Adjust parameters on the left and click predict to generate AI assessment.</p>
            </div>
            """, unsafe_allow_html=True)
