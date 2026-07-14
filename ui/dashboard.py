import streamlit as st
import pandas as pd
from ui.charts import create_risk_pie_chart, create_age_glucose_scatter, create_correlation_heatmap
from ui.components import render_kpi_card

def render_dashboard(data: pd.DataFrame, clf_acc: float):
    """
    Renders the clinical dashboard overview.
    """
    st.markdown("<h2 class='animate-fade-in'>Clinical Dashboard Overview</h2>", unsafe_allow_html=True)
    
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Avg Glucose", f"{data['Glucose'].mean():.1f}", "mg/dL"),
        ("High Risk Cases", f"{(data['Outcome'] == 1).sum()}", "patients"),
        ("Model Accuracy", f"{clf_acc*100:.1f}%", "validation"),
        ("Total Records", f"{len(data)}", "entries")
    ]
    
    for (title, val, unit), col in zip(kpis, [k1, k2, k3, k4]):
        with col:
            render_kpi_card(title, val, unit)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Risk Distribution</h4>", unsafe_allow_html=True)
        fig_pie = create_risk_pie_chart(data)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Age vs Glucose Distribution</h4>", unsafe_allow_html=True)
        fig_scatter = create_age_glucose_scatter(data)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h4>Feature Correlation Heatmap</h4>", unsafe_allow_html=True)
    fig_corr = create_correlation_heatmap(data)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
