# 🩸 Glucore - Premium AI Healthcare (Glucose Monitoring Prediction)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.20.0%2B-red)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0.0%2B-orange)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

> **Empower your clinical decisions with high-precision machine learning. Real-time predictions, continuous risk assessment, and personalized insights.**

A professional, production-grade Streamlit application that uses Machine Learning (Random Forest) to predict glucose saturation levels and assess the risk of diabetes based on patient physiological parameters (e.g., BMI, Blood Pressure, Insulin, Age). Designed with a premium "Apple/Google Health" glassmorphism aesthetic, featuring micro-animations, dynamic responsive layouts, and robust modular architecture.

## ✨ Features

- **Real-Time Prediction Engine**: Uses Scikit-Learn `RandomForestRegressor` and `RandomForestClassifier` trained on the Pima Indians Diabetes Dataset.
- **Premium UI/UX**: Custom CSS injection overriding Streamlit defaults, featuring glassmorphism cards, floating 3D elements, and animated gradients.
- **Interactive Analytics Dashboard**: Beautiful Plotly charts (Gauge, Heatmaps, Scatter) with custom transparent backgrounds.
- **AI Health Insights**: Dynamic generation of feature importance and model reliability metrics.
- **Patient History Database**: Searchable, gradient-styled dataset explorer.
- **Production-Ready Architecture**: Cleanly separated UI components, prediction logic, configuration, and CSS styles.

---

## 📁 Repository Structure

```
Glucose-Monitoring-Prediction/
├── app.py                      # Application entry point
├── requirements.txt            # Locked dependencies
├── setup.py                    # Package setup
├── pyproject.toml              # Build configuration
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
│
├── assets/                     # Static assets (images, logos)
├── data/                       # Datasets
├── models/                     # Saved models (for future persistence)
├── notebooks/                  # Jupyter notebooks for EDA and model tuning
│   └── Glucose_Monitoring_Prediction.ipynb
│
├── src/                        # Core backend logic
│   └── prediction.py           # Model loading, training, and inference logic
│
├── ui/                         # Frontend UI components
│   ├── charts.py               # Plotly figure generation
│   ├── components.py           # Reusable UI elements (KPIs, buttons)
│   ├── dashboard.py            # Analytics dashboard logic
│   ├── prediction_page.py      # Main prediction page layout
│   ├── sidebar.py              # Navigation and profile
│   └── styles.py               # CSS injection logic
│
├── css/                        # Stylesheets
│   └── main.css                # Global styles, glassmorphism, animations
│
├── tests/                      # Unit tests stub directory
├── docs/                       # Documentation and screenshots
├── .streamlit/                 # Streamlit configuration
│   └── config.toml
└── .github/workflows/          # GitHub Actions CI/CD pipelines
    └── streamlit.yml
```

---

## 🚀 Installation & Local Setup

### Prerequisites
- Python 3.8 or higher.

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/Glucose-Monitoring-Prediction.git
   cd Glucose-Monitoring-Prediction
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python -m streamlit run app.py
   ```

---

## 🌐 Streamlit Deployment

This repository is fully optimized for **Streamlit Community Cloud**.

1. Push your repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click "New app".
4. Select this repository and branch.
5. Set the Main file path to `app.py`.
6. Deploy! The custom `.streamlit/config.toml` and CSS will automatically apply.

---

## 📸 Screenshots (Placeholders)

*Add your high-resolution screenshots here once deployed.*

- **Prediction Page**: `docs/screenshots/prediction_page.png`
- **Dashboard Analytics**: `docs/screenshots/dashboard.png`
- **AI Insights**: `docs/screenshots/insights.png`

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit, Custom HTML/CSS, Plotly
- **Backend/ML**: Python, Pandas, NumPy, Scikit-Learn (Random Forest)
- **Tooling**: GitHub Actions (CI), setuptools

---

## 🔭 Future Scope

- **Model Persistence**: Serialize models as `.pkl` files inside `/models/` to speed up initial app load times.
- **Database Integration**: Connect to PostgreSQL/Firebase to store real patient histories instead of the training dataset.
- **User Authentication**: Implement login flow for doctors.
- **REST API**: Expose the model inference via FastAPI.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@YourUsername](https://github.com/YourUsername)
- LinkedIn: [Your Profile](https://linkedin.com)

---
*If you liked this repository, please leave a ⭐!*
