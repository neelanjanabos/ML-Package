"""
Model Training & Evaluation Script for Network Attack Severity Prediction System.
Trains 5 Machine Learning algorithms on the UNSW-NB15 dataset:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (GaussianNB)
5. Support Vector Machine (Linear SVM with probability calibration)

Saves trained models, preprocessing artifacts, and performance metrics.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessing import (
    load_data, 
    preprocess_features, 
    FEATURE_COLUMNS, 
    NUMERICAL_FEATURES, 
    CATEGORICAL_FEATURES,
    SEVERITY_CLASSES
)

def train_and_evaluate():
    print("=" * 70)
    print("NETWORK ATTACK SEVERITY PREDICTION SYSTEM - MODEL TRAINING")
    print("=" * 70)
    
    os.makedirs('models', exist_ok=True)
    
    # 1. Load dataset
    data_path = os.path.join('data', 'dataset.csv')
    print(f"\n[1/6] Loading dataset from '{data_path}'...")
    df = load_data(data_path)
    print(f"      Total records loaded: {len(df):,}")
    print(f"      Total features: {len(FEATURE_COLUMNS)}")
    print(f"      Severity distribution:\n{df['severity'].value_counts().to_string()}")
    
    # 2. Train / Test Split (Avoids Data Leakage)
    print("\n[2/6] Splitting data into training (75%) and testing (25%) sets...")
    X_raw = df[FEATURE_COLUMNS].copy()
    y_severity = df['severity'].copy()
    y_attack = df['attack_cat'].copy()
    
    X_train_raw, X_test_raw, y_train, y_test, y_attack_train, y_attack_test = train_test_split(
        X_raw, y_severity, y_attack,
        test_size=0.25,
        stratify=y_severity,
        random_state=42
    )
    print(f"      Training samples: {len(X_train_raw):,}")
    print(f"      Testing samples:  {len(X_test_raw):,}")
    
    # 3. Preprocess Features (Fit on train, transform test)
    print("\n[3/6] Preprocessing features (Categorical LabelEncoding & StandardScaler)...")
    X_train, encoders, scaler = preprocess_features(X_train_raw, is_training=True)
    X_test, _, _ = preprocess_features(X_test_raw, encoders=encoders, scaler=scaler, is_training=False)
    
    # 4. Define the 5 Models
    # SVM is wrapped in CalibratedClassifierCV to enable probability / confidence estimation
    model_definitions = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=12, min_samples_split=5, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Naive Bayes': GaussianNB(),
        'SVM': CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000, dual='auto'))
    }
    
    filename_map = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'KNN': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'SVM': 'svm.pkl'
    }
    
    # 5. Train & Evaluate Models
    print("\n[4/6] Training and evaluating the 5 Machine Learning algorithms...")
    metrics_report = {}
    trained_models = {}
    
    print("-" * 85)
    print(f"{'Algorithm':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Time (s)':<8}")
    print("-" * 85)
    
    for name, model in model_definitions.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        train_duration = round(time.time() - start_time, 2)
        
        # Predict on holdout test set
        y_pred = model.predict(X_test)
        
        # Calculate standard classification metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        # Compute confusion matrix with standard order: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        cm = confusion_matrix(y_test, y_pred, labels=SEVERITY_CLASSES).tolist()
        
        # Check if probability output is available
        has_proba = hasattr(model, 'predict_proba')
        
        metrics_report[name] = {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'train_time': train_duration,
            'confusion_matrix': cm,
            'classes': SEVERITY_CLASSES,
            'has_proba': has_proba
        }
        
        trained_models[name] = model
        
        print(f"{name:<22} | {acc*100:>8.2f}% | {prec*100:>8.2f}% | {rec*100:>8.2f}% | {f1*100:>8.2f}% | {train_duration:>6.2f}s")
    print("-" * 85)
    
    # Train auxiliary attack type classifier for multi-class attack identification
    print("\n[5/6] Training complementary Attack Type Classifier (Decision Tree)...")
    attack_classifier = DecisionTreeClassifier(max_depth=12, random_state=42)
    attack_classifier.fit(X_train, y_attack_train)
    attack_acc = accuracy_score(y_attack_test, attack_classifier.predict(X_test))
    print(f"      Attack Category Identification Accuracy: {attack_acc * 100:.2f}%")
    
    # 6. Save Models, Encoders, Scaler, and Metrics
    print("\n[6/6] Saving trained models and preprocessing artifacts to 'models/'...")
    for name, model in trained_models.items():
        joblib.dump(model, os.path.join('models', filename_map[name]))
        
    joblib.dump(attack_classifier, os.path.join('models', 'attack_classifier.pkl'))
    joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
    joblib.dump(encoders, os.path.join('models', 'encoders.pkl'))
    
    # Identify Best Model based on F1-Score
    best_model_name = max(metrics_report.keys(), key=lambda k: metrics_report[k]['f1_score'])
    
    summary_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_records': len(df),
        'features_count': len(FEATURE_COLUMNS),
        'train_samples': len(X_train_raw),
        'test_samples': len(X_test_raw),
        'best_model': best_model_name,
        'best_accuracy': metrics_report[best_model_name]['accuracy'],
        'best_f1': metrics_report[best_model_name]['f1_score'],
        'models': metrics_report,
        'attack_categories': df['attack_cat'].value_counts().to_dict(),
        'severity_distribution': df['severity'].value_counts().to_dict()
    }
    
    with open(os.path.join('models', 'metrics.json'), 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"\nTraining Complete! Best Performing Model: {best_model_name} (Accuracy: {summary_data['best_accuracy']}%, F1: {summary_data['best_f1']}%)")
    print(f"Metrics saved to 'models/metrics.json'. Ready for Flask web application.")
    return summary_data

if __name__ == '__main__':
    train_and_evaluate()
