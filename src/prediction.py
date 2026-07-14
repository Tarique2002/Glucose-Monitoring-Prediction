import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from typing import Tuple, List, Dict, Any

@st.cache_data
def load_and_train() -> Tuple[pd.DataFrame, RandomForestRegressor, RandomForestClassifier, StandardScaler, StandardScaler, float, float, float, pd.DataFrame, List[str]]:
    """
    Load dataset from remote URL, preprocess, and train the prediction models.
    
    Returns:
        Tuple containing the dataset, trained models, scalers, evaluation metrics, feature importance, and feature names.
    """
    url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
    data = pd.read_csv(url)

    # --- Regression Model (Glucose Prediction) ---
    reg_features = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                     'DiabetesPedigreeFunction', 'Age']
    X_reg = data[reg_features]
    y_reg = data['Glucose']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    scaler_reg = StandardScaler()
    X_train_r_scaled = scaler_reg.fit_transform(X_train_r)
    X_test_r_scaled = scaler_reg.transform(X_test_r)

    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_r_scaled, y_train_r)

    y_pred_r = reg_model.predict(X_test_r_scaled)
    reg_mae = mean_absolute_error(y_test_r, y_pred_r)
    reg_r2 = r2_score(y_test_r, y_pred_r)

    # --- Classification Model (Diabetes Risk) ---
    clf_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age']
    X_clf = data[clf_features]
    y_clf = data['Outcome']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    scaler_clf = StandardScaler()
    X_train_c_scaled = scaler_clf.fit_transform(X_train_c)
    X_test_c_scaled = scaler_clf.transform(X_test_c)

    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train_c_scaled, y_train_c)

    y_pred_c = clf_model.predict(X_test_c_scaled)
    clf_acc = accuracy_score(y_test_c, y_pred_c)

    # Feature importance
    feature_imp = pd.DataFrame({
        'Feature': reg_features,
        'Importance': reg_model.feature_importances_
    }).sort_values('Importance', ascending=True)

    return data, reg_model, clf_model, scaler_reg, scaler_clf, reg_mae, reg_r2, clf_acc, feature_imp, reg_features

def predict_patient(
    inputs: Dict[str, float],
    reg_model: RandomForestRegressor,
    clf_model: RandomForestClassifier,
    scaler_reg: StandardScaler,
    scaler_clf: StandardScaler
) -> Dict[str, Any]:
    """
    Run prediction on new patient data.
    """
    # Glucose prediction
    input_reg = np.array([[
        inputs['pregnancies'], 
        inputs['blood_pressure'], 
        inputs['skin_thickness'], 
        inputs['insulin'], 
        inputs['bmi'], 
        inputs['dpf'], 
        inputs['age']
    ]])
    input_reg_scaled = scaler_reg.transform(input_reg)
    predicted_glucose = reg_model.predict(input_reg_scaled)[0]

    # Diabetes risk
    input_clf = np.array([[
        inputs['pregnancies'], 
        predicted_glucose, 
        inputs['blood_pressure'], 
        inputs['skin_thickness'], 
        inputs['insulin'], 
        inputs['bmi'], 
        inputs['dpf'], 
        inputs['age']
    ]])
    input_clf_scaled = scaler_clf.transform(input_clf)
    diabetes_risk = clf_model.predict(input_clf_scaled)[0]
    risk_proba = clf_model.predict_proba(input_clf_scaled)[0]
    
    return {
        'predicted_glucose': predicted_glucose,
        'diabetes_risk': diabetes_risk,
        'risk_proba': risk_proba
    }
