import sqlite3
import json
from datetime import datetime
import pandas as pd

DB_PATH = "predictions_history.db"

def init_db():
    """Initialize the SQLite database for prediction history and user profiles."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        user_id TEXT DEFAULT 'guest',
        age INTEGER,
        bmi REAL,
        glucose REAL,
        predicted_risk INTEGER,
        confidence REAL,
        health_score INTEGER,
        full_inputs TEXT
    )
    ''')
    conn.commit()
    conn.close()

def save_prediction(user_id: str, inputs: dict, results: dict):
    """Save a prediction result to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    glucose = results['predicted_glucose']
    risk = results['diabetes_risk']
    confidence = results['risk_proba'][1] if risk == 1 else results['risk_proba'][0]
    health_score = results.get('health_score', 0)
    
    cursor.execute('''
    INSERT INTO predictions (timestamp, user_id, age, bmi, glucose, predicted_risk, confidence, health_score, full_inputs)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,
        user_id,
        inputs['age'],
        inputs['bmi'],
        glucose,
        int(risk),
        float(confidence),
        int(health_score),
        json.dumps(inputs)
    ))
    
    conn.commit()
    conn.close()

def get_prediction_history(user_id: str = 'guest') -> pd.DataFrame:
    """Retrieve prediction history for a user."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return df
