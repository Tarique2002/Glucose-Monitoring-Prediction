# pyrefly: ignore [missing-import]
import streamlit as st
import os

# Create database if not exists
from src.database import init_db
if not os.path.exists("predictions_history.db"):
    init_db()

from ui.styles import inject_custom_css
from ui.sidebar import render_sidebar
from ui.prediction_page import render_prediction_page
from ui.dashboard import render_dashboard
from ui.charts import create_feature_importance_chart
from ui.components import render_ai_assistant_button
from src.prediction import load_and_train
from ui.explainability import render_explainability_page
from ui.simulator import render_simulator
from ui.chatbot import render_chatbot
from ui.model_performance import render_model_performance
from ui.history import render_history

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Health Pro | Portfolio Showcase",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS INJECTION ---
inject_custom_css()

# --- MODEL LOADING & TRAINING ---
# Loads models once and caches them
model_data = load_and_train()

# --- SIDEBAR NAVIGATION ---
menu = render_sidebar()





# --- MAIN APP ROUTING ---
if menu == "Home & Predict":
    render_prediction_page(
        model_data['reg_model'], 
        model_data['primary_clf'], 
        model_data['scaler_reg'], 
        model_data['scaler_clf']
    )

elif menu == "Lifestyle Simulator":
    render_simulator(
        model_data['reg_model'], 
        model_data['primary_clf'], 
        model_data['scaler_reg'], 
        model_data['scaler_clf']
    )

elif menu == "Explainability (SHAP)":
    render_explainability_page(
        model_data['primary_clf'], 
        model_data['X_test_c_scaled'], 
        model_data['scaler_clf']
    )

elif menu == "Analytics Dashboard":
    render_dashboard(model_data['data'], model_data['clf_acc'])

elif menu == "Model Performance":
    render_model_performance(
        model_data['model_metrics'], 
        model_data['primary_clf'], 
        model_data['X_test_c_scaled'], 
        model_data['y_test_c']
    )

elif menu == "Prediction History":
    render_history()

elif menu == "AI Assistant":
    render_chatbot()

# --- FLOATING AI ASSISTANT ---
render_ai_assistant_button()
