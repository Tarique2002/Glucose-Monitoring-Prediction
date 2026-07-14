import streamlit as st
import shap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def create_shap_waterfall(shap_values, feature_names, base_value, max_display=10):
    """Create a Plotly waterfall chart for SHAP values of a single prediction."""
    # Ensure inputs are correctly shaped
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # For classification, take positive class
        
    vals = shap_values
    
    # Sort features by absolute SHAP value
    feature_order = np.argsort(np.abs(vals))
    
    # Take top N
    top_indices = feature_order[-max_display:]
    
    names = [feature_names[i] for i in top_indices]
    values = [vals[i] for i in top_indices]
    
    # Calculate "Other" if there are more features
    if len(vals) > max_display:
        other_sum = np.sum([vals[i] for i in feature_order[:-max_display]])
        names = ["Other features"] + names
        values = [other_sum] + values
        
    # Prepare data for waterfall
    measure = ["relative"] * len(values)
    measure.append("total")
    
    names.append("Output")
    values.append(base_value + sum(values))
    
    # Add base value as the first step
    measure = ["absolute"] + measure
    names = ["Base Value"] + names
    values = [base_value] + values

    fig = go.Figure(go.Waterfall(
        name = "SHAP", orientation = "v",
        measure = measure,
        x = names,
        textposition = "outside",
        text = [f"{v:+.2f}" for v in values],
        y = values,
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title = "Local Explanation (Waterfall)",
        showlegend = False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def render_explainability_page(model, X_train, scaler, patient_inputs=None):
    st.markdown("<h2 class='animate-fade-in'>🔍 AI Explainability (SHAP)</h2>", unsafe_allow_html=True)
    st.markdown("Understand *why* the AI makes specific predictions.")
    
    with st.spinner("Calculating SHAP values (this may take a moment)..."):
        # We need a background dataset for SHAP
        # Using a sample of X_train to speed up TreeExplainer
        background = shap.sample(X_train, 100)
        explainer = shap.TreeExplainer(model)
        
        # If we have a specific patient to explain
        if patient_inputs is not None:
            st.markdown("### Local Explanation (Current Patient)")
            
            # Extract SHAP values for this specific instance
            shap_vals = explainer.shap_values(patient_inputs)
            expected_value = explainer.expected_value
            
            if isinstance(expected_value, list) or isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1] # binary classification
                
            fig_waterfall = create_shap_waterfall(
                shap_vals[0] if isinstance(shap_vals, list) else shap_vals, 
                ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age'],
                expected_value
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
        st.markdown("### Global Explanation (Model Behavior)")
        st.info("Global SHAP summary shows how each feature impacts the model across all patients.")
        
        # Calculate SHAP for the background dataset
        shap_values_global = explainer.shap_values(background)
        
        # Create a proxy summary bar chart using Plotly
        vals = np.abs(shap_values_global[1] if isinstance(shap_values_global, list) else shap_values_global).mean(0)
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age']
        
        df_shap = pd.DataFrame(list(zip(feature_names, vals)), columns=['Feature', 'Mean |SHAP|'])
        df_shap = df_shap.sort_values(by='Mean |SHAP|', ascending=True)
        
        fig_bar = px.bar(df_shap, x='Mean |SHAP|', y='Feature', orientation='h', title="Global Feature Importance (SHAP)")
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_bar, use_container_width=True)
