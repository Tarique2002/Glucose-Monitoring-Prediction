import streamlit as st
import numpy as np
from ui.charts import create_gauge_chart

def render_simulator(reg_model, clf_model, scaler_reg, scaler_clf):
    st.markdown("<h2 class='animate-fade-in'>🧪 Lifestyle Simulator & What-if Analysis</h2>", unsafe_allow_html=True)
    st.markdown("Adjust your lifestyle metrics to simulate how it affects your predicted risk in real-time.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.container():
            st.markdown("### Base Profile")
            age = st.slider("Age", 18, 90, 45, key="sim_age")
            pregnancies = st.slider("Pregnancies", 0, 17, 1, key="sim_preg")
            dpf = st.slider("Genetics (DPF)", 0.05, 2.50, 0.5, key="sim_dpf")
            
        with st.container():
            st.markdown("### Modifiable Lifestyle Factors")
            st.info("💡 What if you lost weight or lowered blood pressure?")
            bmi = st.slider("BMI (Weight Management)", 15.0, 50.0, 30.0, 0.1, key="sim_bmi")
            bp = st.slider("Blood Pressure", 50, 140, 80, key="sim_bp")
            skin = st.slider("Skin Thickness", 0, 99, 20, key="sim_skin")
            insulin = st.slider("Insulin Levels", 0, 846, 79, key="sim_ins")

    with col2:
        # Glucose prediction
        input_reg = np.array([[pregnancies, bp, skin, insulin, bmi, dpf, age]])
        input_reg_scaled = scaler_reg.transform(input_reg)
        predicted_glucose = reg_model.predict(input_reg_scaled)[0]

        # Diabetes risk
        input_clf = np.array([[pregnancies, predicted_glucose, bp, skin, insulin, bmi, dpf, age]])
        input_clf_scaled = scaler_clf.transform(input_clf)
        diabetes_risk = clf_model.predict(input_clf_scaled)[0]
        risk_proba = clf_model.predict_proba(input_clf_scaled)[0]
        
        is_high_risk = predicted_glucose >= 126 or diabetes_risk == 1
        card_class = "result-danger" if is_high_risk else "result-safe"
        risk_color = "var(--danger)" if is_high_risk else "var(--accent)"
        risk_text = "High Risk" if is_high_risk else "Normal Range"
        
        with st.container():
            st.markdown(f"""
            <div class="result-card {card_class} animate-fade-in">
                <h2 style="color: {risk_color}">{predicted_glucose:.1f} <span>mg/dL</span></h2>
                <h4 style="color: {risk_color};">{risk_text}</h4>
                <p>Simulated Risk Probability: {risk_proba[1]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig_gauge = create_gauge_chart(predicted_glucose, risk_color)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        st.markdown("### Real-Time Insights")
        if bmi > 25:
            st.warning("⚠️ Try reducing the BMI slider to see how weight loss decreases diabetes probability.")
        if bp > 120:
            st.warning("⚠️ Lowering blood pressure significantly improves cardiovascular and metabolic risk.")
        if is_high_risk:
            st.error("This simulated profile is in the danger zone. Immediate lifestyle intervention is recommended.")
        else:
            st.success("This profile looks healthy! Keep maintaining a balanced diet and regular exercise.")
