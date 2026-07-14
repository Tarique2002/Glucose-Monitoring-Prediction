import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve

def render_model_performance(model_metrics: dict, primary_clf, X_test_c_scaled, y_test_c):
    st.markdown("<h2 class='animate-fade-in'>📈 Model Performance & Comparison</h2>", unsafe_allow_html=True)
    st.markdown("Compare the clinical accuracy of various machine learning models trained on the dataset.")
    
    # 1. Model Comparison Table
    st.markdown("### Metrics Comparison")
    df_metrics = pd.DataFrame(model_metrics).T
    # Format as percentages
    df_metrics_pct = (df_metrics * 100).round(2).astype(str) + "%"
    
    st.dataframe(
        df_metrics.style.background_gradient(cmap='viridis', axis=0),
        use_container_width=True
    )
    
    # 2. Bar chart comparison
    st.markdown("### Accuracy Comparison")
    fig_comp = px.bar(
        df_metrics.reset_index(), 
        x='index', y='Accuracy', 
        color='Accuracy',
        labels={'index': 'Model', 'Accuracy': 'Accuracy Score'},
        title="Model Accuracy Comparison"
    )
    fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # 3. ROC Curve & Confusion Matrix for Primary Model
    st.markdown(f"### Deep Dive: Random Forest (Primary Model)")
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        y_pred = primary_clf.predict(X_test_c_scaled)
        cm = confusion_matrix(y_test_c, y_pred)
        
        fig_cm = px.imshow(
            cm, 
            text_auto=True, 
            color_continuous_scale='Blues',
            labels=dict(x="Predicted Label", y="True Label"),
            x=['Normal', 'Diabetic'],
            y=['Normal', 'Diabetic'],
            title="Confusion Matrix"
        )
        fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col2:
        # ROC Curve
        y_proba = primary_clf.predict_proba(X_test_c_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test_c, y_proba)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC Curve", mode='lines', line=dict(color='blue', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Guess", mode='lines', line=dict(color='gray', dash='dash')))
        
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_roc, use_container_width=True)
