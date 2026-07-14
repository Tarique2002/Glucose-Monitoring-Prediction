import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Tuple, List, Dict, Any

@st.cache_data
def load_and_train():
    """
    Load dataset, train multiple prediction models, and calculate evaluation metrics.
    """
    url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
    data = pd.read_csv(url)
    
    # 1. Data Quality Checks & Automated Cleaning
    missing_vals_start = data.isnull().sum().sum()
    outlier_warnings = []
    
    # In Pima Indians dataset, 0 means missing for these columns
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Count how many 0s exist
    zero_count = (data[zero_cols] == 0).sum().sum()
    
    if zero_count > 0:
        # Replace 0s with NaN
        data[zero_cols] = data[zero_cols].replace(0, np.nan)
        # Impute with column median
        data[zero_cols] = data[zero_cols].fillna(data[zero_cols].median())
        outlier_warnings.append(f"AI Data Cleaning: Automatically imputed {zero_count} missing/zero values using median interpolation.")
        
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

    # --- Classification Models (Diabetes Risk) ---
    clf_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age']
    X_clf = data[clf_features]
    y_clf = data['Outcome']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    scaler_clf = StandardScaler()
    X_train_c_scaled = scaler_clf.fit_transform(X_train_c)
    X_test_c_scaled = scaler_clf.transform(X_test_c)

    # Dictionary of classifiers to compare
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    model_metrics = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train_c_scaled, y_train_c)
        trained_models[name] = model
        
        y_pred = model.predict(X_test_c_scaled)
        y_proba = model.predict_proba(X_test_c_scaled)[:, 1]
        
        model_metrics[name] = {
            "Accuracy": accuracy_score(y_test_c, y_pred),
            "Precision": precision_score(y_test_c, y_pred),
            "Recall": recall_score(y_test_c, y_pred),
            "F1": f1_score(y_test_c, y_pred),
            "ROC AUC": roc_auc_score(y_test_c, y_proba)
        }

    # Primary classifier for general usage is Random Forest
    primary_clf = trained_models["Random Forest"]
    clf_acc = model_metrics["Random Forest"]["Accuracy"]

    # Feature importance
    feature_imp = pd.DataFrame({
        'Feature': clf_features,
        'Importance': primary_clf.feature_importances_
    }).sort_values('Importance', ascending=True)

    return {
        'data': data,
        'reg_model': reg_model,
        'primary_clf': primary_clf,
        'trained_models': trained_models,
        'scaler_reg': scaler_reg,
        'scaler_clf': scaler_clf,
        'reg_mae': reg_mae,
        'reg_r2': reg_r2,
        'clf_acc': clf_acc,
        'feature_imp': feature_imp,
        'reg_features': reg_features,
        'clf_features': clf_features,
        'model_metrics': model_metrics,
        'missing_vals': missing_vals_start,
        'outlier_warnings': outlier_warnings,
        'X_test_c_scaled': X_test_c_scaled,
        'y_test_c': y_test_c
    }

def calculate_health_score(predicted_glucose: float, bmi: float, blood_pressure: float) -> int:
    """Calculate a 0-100 health score based on key metrics."""
    score = 100
    
    # Glucose penalty
    if predicted_glucose > 140:
        score -= min(40, (predicted_glucose - 140) * 0.5)
    elif predicted_glucose < 70:
        score -= 20
        
    # BMI penalty
    if bmi > 25:
        score -= min(20, (bmi - 25) * 1.5)
    elif bmi < 18.5:
        score -= 10
        
    # BP Penalty
    if blood_pressure > 120:
        score -= min(20, (blood_pressure - 120) * 0.5)
        
    return max(0, min(100, int(score)))

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
    
    health_score = calculate_health_score(predicted_glucose, inputs['bmi'], inputs['blood_pressure'])
    
    return {
        'predicted_glucose': predicted_glucose,
        'diabetes_risk': diabetes_risk,
        'risk_proba': risk_proba,
        'health_score': health_score,
        'input_clf_scaled': input_clf_scaled # needed for SHAP
    }
