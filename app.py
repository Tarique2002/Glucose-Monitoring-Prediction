import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Glucose Monitoring Prediction",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6366F1;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #4338CA;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #A78BFA;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
    }
    .prediction-box {
        background: linear-gradient(135deg, #065F46 0%, #047857 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #10B981;
        margin: 1rem 0;
    }
    .prediction-box.high-risk {
        background: linear-gradient(135deg, #7F1D1D 0%, #991B1B 100%);
        border: 1px solid #EF4444;
    }
    .stApp {
        background-color: #0F172A;
    }
    div[data-testid="stSidebar"] {
        background-color: #1E293B;
    }
    .footer {
        text-align: center;
        color: #64748B;
        padding: 2rem 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_train():
    """Load dataset and train models"""
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


data, reg_model, clf_model, scaler_reg, scaler_clf, reg_mae, reg_r2, clf_acc, feature_imp, reg_features = load_and_train()

# Header
st.markdown('<p class="main-header">🩸 Glucose Monitoring Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-powered glucose level prediction and diabetes risk assessment using the Pima Indians Diabetes Dataset</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🧑‍⚕️ Patient Input")
st.sidebar.markdown("Enter patient health parameters below:")

pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 20, 130, 72)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 29)
insulin = st.sidebar.slider("Insulin (mu U/ml)", 0, 850, 80)
bmi = st.sidebar.slider("BMI", 10.0, 70.0, 26.0, 0.1)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.05, 2.50, 0.47, 0.01)
age = st.sidebar.slider("Age", 18, 90, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by [Mohd. Tarique Ansari](https://tarique2002.github.io)**")
st.sidebar.markdown("[GitHub Repo](https://github.com/Tarique2002/Glucose-Monitoring-Prediction)")

# Predict
if st.sidebar.button("🔍 Predict", use_container_width=True):
    # Glucose prediction
    input_reg = np.array([[pregnancies, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_reg_scaled = scaler_reg.transform(input_reg)
    predicted_glucose = reg_model.predict(input_reg_scaled)[0]

    # Diabetes risk
    input_clf = np.array([[pregnancies, predicted_glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_clf_scaled = scaler_clf.transform(input_clf)
    diabetes_risk = clf_model.predict(input_clf_scaled)[0]
    risk_proba = clf_model.predict_proba(input_clf_scaled)[0]

    # Results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{predicted_glucose:.1f} mg/dL</div>
            <div class="metric-label">Predicted Glucose Level</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        risk_text = "High Risk" if diabetes_risk == 1 else "Low Risk"
        risk_color = "#EF4444" if diabetes_risk == 1 else "#10B981"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {risk_color}">{risk_text}</div>
            <div class="metric-label">Diabetes Risk Assessment</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{risk_proba[1]*100:.1f}%</div>
            <div class="metric-label">Risk Probability</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Glucose level interpretation
    if predicted_glucose < 70:
        level, color, emoji = "Low (Hypoglycemia)", "#F59E0B", "⚠️"
        advice = "Glucose level is below normal. Recommend consuming fast-acting carbohydrates and consulting a healthcare provider."
    elif predicted_glucose < 100:
        level, color, emoji = "Normal", "#10B981", "✅"
        advice = "Glucose level is within the normal fasting range. Maintain healthy lifestyle habits."
    elif predicted_glucose < 126:
        level, color, emoji = "Pre-Diabetic", "#F59E0B", "⚠️"
        advice = "Glucose level indicates pre-diabetes. Recommend lifestyle modifications, regular exercise, and dietary changes."
    else:
        level, color, emoji = "Diabetic Range", "#EF4444", "🚨"
        advice = "Glucose level is in the diabetic range. Strongly recommend consulting an endocrinologist for proper diagnosis and treatment."

    st.markdown(f"""
    <div class="prediction-box {'high-risk' if predicted_glucose >= 126 else ''}">
        <h2 style="color: white; margin: 0;">{emoji} {level}</h2>
        <p style="color: #E2E8F0; margin: 0.5rem 0 0 0;">{advice}</p>
    </div>
    """, unsafe_allow_html=True)

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_glucose,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Glucose Level (mg/dL)", 'font': {'size': 20, 'color': 'white'}},
        number={'font': {'size': 40, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 250], 'tickwidth': 1, 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
            'bar': {'color': color},
            'bgcolor': '#1E293B',
            'borderwidth': 2,
            'bordercolor': '#334155',
            'steps': [
                {'range': [0, 70], 'color': '#1E3A5F'},
                {'range': [70, 100], 'color': '#064E3B'},
                {'range': [100, 126], 'color': '#78350F'},
                {'range': [126, 250], 'color': '#7F1D1D'}
            ],
            'threshold': {
                'line': {'color': '#F43F5E', 'width': 4},
                'thickness': 0.75,
                'value': predicted_glucose
            }
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='#0F172A',
        font={'color': 'white'},
        height=350,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

else:
    st.info("👈 Adjust patient parameters in the sidebar and click **Predict** to see results.")

# Dataset Explorer & Model Performance
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Dataset Explorer", "📈 Model Performance", "🔬 Feature Importance"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(data, x='Glucose', nbins=30, color_discrete_sequence=['#6366F1'],
                                 title='Distribution of Glucose Levels')
        fig_hist.update_layout(paper_bgcolor='#0F172A', plot_bgcolor='#1E293B',
                                font_color='white', title_font_color='#A78BFA')
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        fig_scatter = px.scatter(data, x='BMI', y='Glucose', color='Outcome',
                                  color_discrete_map={0: '#6366F1', 1: '#F43F5E'},
                                  title='BMI vs Glucose by Diabetes Outcome',
                                  labels={'Outcome': 'Diabetic'})
        fig_scatter.update_layout(paper_bgcolor='#0F172A', plot_bgcolor='#1E293B',
                                   font_color='white', title_font_color='#A78BFA')
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.dataframe(data.head(20), use_container_width=True)

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Regression MAE", f"{reg_mae:.2f} mg/dL")
    with col2:
        st.metric("Regression R² Score", f"{reg_r2:.4f}")
    with col3:
        st.metric("Classification Accuracy", f"{clf_acc*100:.1f}%")

    fig_corr = px.imshow(data.corr(numeric_only=True), text_auto='.2f',
                          color_continuous_scale='RdBu_r', title='Feature Correlation Heatmap')
    fig_corr.update_layout(paper_bgcolor='#0F172A', plot_bgcolor='#1E293B',
                            font_color='white', title_font_color='#A78BFA', height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    fig_imp = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale='Viridis',
                      title='Feature Importance (Random Forest)')
    fig_imp.update_layout(paper_bgcolor='#0F172A', plot_bgcolor='#1E293B',
                           font_color='white', title_font_color='#A78BFA', height=400)
    st.plotly_chart(fig_imp, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    Built with ❤️ by <a href="https://tarique2002.github.io" style="color: #6366F1;">Mohd. Tarique Ansari</a> |
    <a href="https://github.com/Tarique2002/Glucose-Monitoring-Prediction" style="color: #6366F1;">GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
