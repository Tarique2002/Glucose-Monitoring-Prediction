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
    # Hero Section using Streamlit layout
    col_hero1, col_hero2 = st.columns([2, 1])
    with col_hero1:
        st.markdown("""
        <div class="hero-text animate-fade-in">
            <div class="hero-badge">✨ AI-Powered Healthcare</div>
            <h1>Next-Gen Glucose<br>Monitoring</h1>
            <p>Empower your clinical decisions with high-precision machine learning. Real-time predictions, continuous risk assessment, and personalized insights.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_hero2:
        st.markdown('<div class="hero-img floating-3d animate-fade-in">🧬</div>', unsafe_allow_html=True)

    st.markdown("<h2 style='margin-top: 2rem; margin-bottom: 1.5rem;'>Patient Parameters</h2>", unsafe_allow_html=True)
    
    form_col1, form_col2 = st.columns([1, 1.5])
    
    with form_col1:
        with st.container():
            st.markdown("<h4 style='color: var(--primary);'>📋 Medical History</h4>", unsafe_allow_html=True)
            pregnancies = st.slider("Pregnancies", 0, 17, 1)
            age = st.slider("Age", 18, 90, 30)
            bmi = st.slider("BMI", 10.0, 70.0, 26.0, 0.1)
            
            st.markdown("<br><h4 style='color: var(--accent);'>🔬 Vitals & Labs</h4>", unsafe_allow_html=True)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 20, 130, 72)
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 29)
            insulin = st.slider("Insulin (mu U/ml)", 0, 850, 80)
            dpf = st.slider("Diabetes Pedigree Function", 0.05, 2.50, 0.47, 0.01)
            
        predict_btn = st.button("Generate AI Prediction", type="primary")

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
                
                with st.container():
                    st.markdown(f"""
                    <div class="result-card {card_class} animate-fade-in">
                        <div class="result-icon floating-3d">{'🚨' if is_high_risk else '✅'}</div>
                        <h2 style="color: {risk_color}">{predicted_glucose:.1f} <span>mg/dL</span></h2>
                        <h4 style="color: {risk_color};">{risk_text}</h4>
                        <p>Confidence Score: {confidence_score:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    r_col1, r_col2, r_col3 = st.columns(3)
                    with r_col1:
                        st.info(f"🥗 **Diet:**\n\n{'Strict low GI foods' if is_high_risk else 'Maintain balanced diet'}")
                    with r_col2:
                        st.success(f"🏃 **Activity:**\n\n{'Daily cardio required' if is_high_risk else 'Regular exercise'}")
                    with r_col3:
                        st.warning(f"💊 **Action:**\n\n{'Consult physician' if is_high_risk else 'Routine checkup'}")
                
                # Gauge Chart
                fig_gauge = create_gauge_chart(predicted_glucose, risk_color)
                st.plotly_chart(fig_gauge, use_container_width=True)

        else:
            # Idle State for right column
            with st.container():
                st.markdown("""
                <div class="idle-state">
                    <div class="idle-icon floating-3d">🩺</div>
                    <h3>Awaiting Input</h3>
                    <p>Adjust parameters on the left and click predict to generate AI assessment.</p>
                </div>
                """, unsafe_allow_html=True)
