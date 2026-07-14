import streamlit as st
from src.database import get_prediction_history

def render_history():
    st.markdown("<h2 class='animate-fade-in'>🕒 Patient Prediction History</h2>", unsafe_allow_html=True)
    st.markdown("A timeline of past predictions saved locally.")
    
    df_history = get_prediction_history(user_id='guest')
    
    if df_history.empty:
        st.info("No prediction history found. Run a prediction on the Home page to see it here!")
        return
        
    st.markdown(f"**Total records:** {len(df_history)}")
    
    # Timeline
    for _, row in df_history.iterrows():
        with st.container():
            is_high_risk = row['predicted_risk'] == 1
            color = "var(--danger)" if is_high_risk else "var(--accent)"
            icon = "🚨" if is_high_risk else "✅"
            
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding: 15px; margin-bottom: 15px; background: rgba(255,255,255,0.5); border-radius: 0 10px 10px 0;">
                <div style="display:flex; justify-content: space-between;">
                    <div style="font-size: 1.1rem; font-weight: bold;">{icon} Predicted Glucose: {row['glucose']:.1f} mg/dL</div>
                    <div style="color: var(--text-muted); font-size: 0.8rem;">{row['timestamp']}</div>
                </div>
                <div style="margin-top: 8px; display: flex; gap: 20px; font-size: 0.9rem;">
                    <div><b>Health Score:</b> {row['health_score']}/100</div>
                    <div><b>Confidence:</b> {row['confidence']*100:.1f}%</div>
                    <div><b>BMI:</b> {row['bmi']}</div>
                    <div><b>Age:</b> {row['age']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    # Export capability
    st.markdown("### Export Data")
    csv = df_history.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name='prediction_history.csv',
        mime='text/csv',
    )
