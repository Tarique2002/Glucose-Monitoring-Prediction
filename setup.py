from setuptools import setup, find_packages

setup(
    name="glucose-monitoring-prediction",
    version="1.0.0",
    description="A professional AI Healthcare Streamlit application for predicting glucose levels and diabetes risk.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.20.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "plotly>=5.9.0",
    ],
    python_requires=">=3.8",
)
