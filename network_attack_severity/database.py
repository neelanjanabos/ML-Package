"""
Database Management Module for Network Attack Severity Prediction System.
Uses standard Python SQLite to store prediction logs and history.
Simple, light, zero-configuration database suitable for a college student project.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'predictions.db')

def get_db_connection():
    """Returns a connection to the SQLite database with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the prediction logs table if it does not already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_used TEXT NOT NULL,
            predicted_severity TEXT NOT NULL,
            predicted_attack TEXT NOT NULL,
            confidence REAL,
            proto TEXT,
            service TEXT,
            state TEXT,
            dur REAL,
            rate REAL,
            sbytes REAL,
            dbytes REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(model_used, severity, attack, confidence, features=None):
    """Logs a single prediction record into SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    features = features or {}
    
    proto = str(features.get('proto', '-'))
    service = str(features.get('service', '-'))
    state = str(features.get('state', '-'))
    dur = float(features.get('dur', 0.0))
    rate = float(features.get('rate', 0.0))
    sbytes = float(features.get('sbytes', 0.0))
    dbytes = float(features.get('dbytes', 0.0))
    
    cursor.execute('''
        INSERT INTO prediction_history 
        (timestamp, model_used, predicted_severity, predicted_attack, confidence, proto, service, state, dur, rate, sbytes, dbytes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, model_used, severity, attack, confidence, proto, service, state, dur, rate, sbytes, dbytes))
    
    conn.commit()
    conn.close()

def get_prediction_history(limit=50):
    """Retrieves the most recent predictions from SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM prediction_history 
        ORDER BY id DESC 
        LIMIT ?
    ''', (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_history_summary():
    """Computes total count, severity distribution, and attack distribution from stored history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as total FROM prediction_history')
    total = cursor.fetchone()['total']
    
    cursor.execute('''
        SELECT predicted_severity, COUNT(*) as cnt 
        FROM prediction_history 
        GROUP BY predicted_severity
    ''')
    severity_dist = {row['predicted_severity']: row['cnt'] for row in cursor.fetchall()}
    
    cursor.execute('''
        SELECT predicted_attack, COUNT(*) as cnt 
        FROM prediction_history 
        GROUP BY predicted_attack
    ''')
    attack_dist = {row['predicted_attack']: row['cnt'] for row in cursor.fetchall()}
    
    cursor.execute('''
        SELECT model_used, COUNT(*) as cnt 
        FROM prediction_history 
        GROUP BY model_used
    ''')
    model_dist = {row['model_used']: row['cnt'] for row in cursor.fetchall()}
    
    conn.close()
    return {
        'total': total,
        'severity_dist': severity_dist,
        'attack_dist': attack_dist,
        'model_dist': model_dist
    }

def clear_prediction_history():
    """Clears the history table for a fresh testing session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM prediction_history')
    conn.commit()
    conn.close()
