# 🩸 Glucose Monitoring Prediction

An ML-powered glucose level prediction and diabetes risk assessment system using the **Pima Indians Diabetes Dataset**.

## 🚀 Live Demo

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://glucose-monitoring-prediction.streamlit.app)

## ✨ Features

- **Glucose Level Prediction** — Random Forest Regressor predicts glucose levels from patient health parameters
- **Diabetes Risk Assessment** — Random Forest Classifier evaluates diabetes risk probability
- **Interactive Dashboard** — Real-time predictions with visual gauge charts
- **Dataset Explorer** — Interactive visualizations of the dataset
- **Model Performance** — View accuracy metrics, correlation heatmaps, and feature importance

## 🛠️ Tech Stack

- **Python** — Core language
- **Streamlit** — Web application framework
- **Scikit-learn** — ML model training (Random Forest)
- **Plotly** — Interactive data visualizations
- **Pandas / NumPy** — Data processing

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| Random Forest Regressor | R² Score | ~0.10 |
| Random Forest Regressor | MAE | ~24 mg/dL |
| Random Forest Classifier | Accuracy | ~76% |

## 🏃 Run Locally

```bash
# Clone the repo
git clone https://github.com/Tarique2002/Glucose-Monitoring-Prediction.git
cd Glucose-Monitoring-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py                                  # Streamlit web application
├── Glucose_Monitoring_Prediction.ipynb     # Jupyter notebook with EDA
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

## 👨‍💻 Author

**Mohd. Tarique Ansari** — [Portfolio](https://tarique2002.github.io) | [LinkedIn](https://www.linkedin.com/in/mohd-tarique-948a37308)
