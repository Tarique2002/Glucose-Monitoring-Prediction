import streamlit as st
import pandas as pd
from ui.charts import (
    create_risk_pie_chart, 
    create_age_glucose_scatter, 
    create_correlation_heatmap,
    create_violin_plot,
    create_box_plot,
    create_radar_chart,
    create_sunburst_chart,
    create_treemap_chart,
    create_histogram
)

def render_dashboard(data: pd.DataFrame, clf_acc: float):
    st.markdown("<h2 class='animate-fade-in'>📊 Advanced Analytics Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("Comprehensive population health analytics derived from the clinical dataset.")
    
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Predictions", len(data), delta="12 today")
    with k2:
        st.metric("Avg Glucose", f"{data['Glucose'].mean():.1f} mg/dL", delta=f"{-1.2}")
    with k3:
        st.metric("Avg BMI", f"{data['BMI'].mean():.1f}", delta=f"{0.4}", delta_color="inverse")
    with k4:
        st.metric("Model Accuracy", f"{clf_acc*100:.1f}%", delta="Validation")
            
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Chart Grid
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Risk Distribution (Sunburst)")
        fig_sun = create_sunburst_chart(data)
        st.plotly_chart(fig_sun, use_container_width=True)
        
    with c2:
        st.markdown("### Age vs Glucose (Scatter)")
        fig_scatter = create_age_glucose_scatter(data)
        st.plotly_chart(fig_scatter, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### Feature Correlation (Heatmap)")
        fig_corr = create_correlation_heatmap(data)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with c4:
        st.markdown("### Population Health Metrics (Radar)")
        fig_radar = create_radar_chart(data)
        st.plotly_chart(fig_radar, use_container_width=True)
        
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("### Glucose Distribution (Violin)")
        fig_violin = create_violin_plot(data, 'Glucose')
        st.plotly_chart(fig_violin, use_container_width=True)
        
    with c6:
        st.markdown("### BMI by Outcome (Box Plot)")
        fig_box = create_box_plot(data, 'BMI', 'Outcome')
        st.plotly_chart(fig_box, use_container_width=True)
        
    st.markdown("### Patient Demographics (Treemap)")
    fig_tree = create_treemap_chart(data)
    st.plotly_chart(fig_tree, use_container_width=True)
    
    st.markdown("### Blood Pressure (Histogram)")
    fig_hist = create_histogram(data, 'BloodPressure')
    st.plotly_chart(fig_hist, use_container_width=True)
