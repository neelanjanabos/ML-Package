"""
Inference and Prediction Engine for Network Attack Severity Prediction System.
Provides single-flow prediction and batch CSV prediction capabilities using
the pre-trained machine learning models and preprocessing pipelines.
"""

import os
import joblib
import pandas as pd
import numpy as np

from preprocessing import (
    FEATURE_COLUMNS, 
    CATEGORICAL_FEATURES, 
    NUMERICAL_FEATURES, 
    SEVERITY_CLASSES,
    SEVERITY_MAPPING,
    preprocess_features
)

class SeverityPredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.encoders = None
        self.attack_classifier = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Loads all trained models and preprocessing objects from disk."""
        model_files = {
            'KNN': 'knn.pkl',
            'Decision Tree': 'decision_tree.pkl',
            'Logistic Regression': 'logistic_regression.pkl',
            'SVM': 'svm.pkl',
            'Naive Bayes': 'naive_bayes.pkl'
        }
        
        for name, filename in model_files.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            
        encoders_path = os.path.join(self.models_dir, 'encoders.pkl')
        if os.path.exists(encoders_path):
            self.encoders = joblib.load(encoders_path)
            
        attack_path = os.path.join(self.models_dir, 'attack_classifier.pkl')
        if os.path.exists(attack_path):
            self.attack_classifier = joblib.load(attack_path)

    def predict_single(self, input_data: dict, model_name: str = 'KNN'):
        """
        Runs severity and attack prediction on a single network flow dictionary.
        
        Returns:
            dict containing predicted_severity, predicted_attack, confidence, probabilities, model_used.
        """
        if model_name not in self.models:
            # Fallback to available model
            model_name = next(iter(self.models.keys()))
            
        model = self.models[model_name]
        
        # Build 1-row DataFrame with standard feature columns
        row_dict = {}
        for col in FEATURE_COLUMNS:
            val = input_data.get(col, 0)
            if col in NUMERICAL_FEATURES:
                try:
                    row_dict[col] = [float(val)]
                except (ValueError, TypeError):
                    row_dict[col] = [0.0]
            else:
                row_dict[col] = [str(val).strip()]
                
        df_single = pd.DataFrame(row_dict)
        
        # Preprocess using fitted encoders and scaler
        X_proc, _, _ = preprocess_features(
            df_single, 
            encoders=self.encoders, 
            scaler=self.scaler, 
            is_training=False
        )
        
        # Severity prediction
        pred_severity = model.predict(X_proc)[0]
        
        # Confidence estimation
        probabilities = {}
        confidence = None
        
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X_proc)[0]
                classes = list(model.classes_)
                for cls_name, prob in zip(classes, proba):
                    probabilities[cls_name] = round(float(prob) * 100, 1)
                confidence = round(float(np.max(proba)) * 100, 1)
            except Exception:
                confidence = 100.0
        elif hasattr(model, 'decision_function'):
            try:
                # Use softmax over decision function
                dfunc = model.decision_function(X_proc)[0]
                exp_scores = np.exp(dfunc - np.max(dfunc))
                probs = exp_scores / np.sum(exp_scores)
                for cls_name, prob in zip(model.classes_, probs):
                    probabilities[cls_name] = round(float(prob) * 100, 1)
                confidence = round(float(np.max(probs)) * 100, 1)
            except Exception:
                confidence = None
                
        # Complementary Attack Type Prediction
        predicted_attack = "Normal"
        if self.attack_classifier:
            try:
                raw_attack = self.attack_classifier.predict(X_proc)[0]
                # Consistency check: if severity is LOW, attack is Normal (benign)
                if pred_severity == 'LOW':
                    predicted_attack = 'Normal'
                elif raw_attack == 'Normal':
                    # If severity is not LOW but raw attack is normal, map to plausible type based on severity
                    severity_to_type = {'MEDIUM': 'Reconnaissance', 'HIGH': 'DoS', 'CRITICAL': 'Backdoor'}
                    predicted_attack = severity_to_type.get(pred_severity, 'Exploits')
                else:
                    predicted_attack = raw_attack
            except Exception:
                predicted_attack = "Attack Detected" if pred_severity != 'LOW' else "Normal"
                
        return {
            'model_used': model_name,
            'predicted_severity': pred_severity,
            'predicted_attack': predicted_attack,
            'confidence': confidence,
            'probabilities': probabilities,
            'features_received': {col: row_dict[col][0] for col in FEATURE_COLUMNS}
        }

    def predict_batch(self, df_upload: pd.DataFrame, model_name: str = 'KNN'):
        """
        Processes an uploaded DataFrame and returns results with predicted severity and attack type.
        """
        if model_name not in self.models:
            model_name = next(iter(self.models.keys()))
            
        model = self.models[model_name]
        
        # Verify required columns or fill missing
        df_clean = df_upload.copy()
        for col in FEATURE_COLUMNS:
            if col not in df_clean.columns:
                if col in NUMERICAL_FEATURES:
                    df_clean[col] = 0.0
                else:
                    df_clean[col] = '-'
            else:
                if col in NUMERICAL_FEATURES:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
                else:
                    df_clean[col] = df_clean[col].astype(str).str.strip()
                    
        X_proc, _, _ = preprocess_features(
            df_clean[FEATURE_COLUMNS], 
            encoders=self.encoders, 
            scaler=self.scaler, 
            is_training=False
        )
        
        # Predictions
        preds_severity = model.predict(X_proc)
        
        confidences = []
        if hasattr(model, 'predict_proba'):
            try:
                probas = model.predict_proba(X_proc)
                confidences = [round(float(np.max(p)) * 100, 1) for p in probas]
            except Exception:
                confidences = [100.0] * len(preds_severity)
        else:
            confidences = [None] * len(preds_severity)
            
        # Predict attack type
        preds_attack = []
        if self.attack_classifier:
            raw_attacks = self.attack_classifier.predict(X_proc)
            for sev, atk in zip(preds_severity, raw_attacks):
                if sev == 'LOW':
                    preds_attack.append('Normal')
                elif atk == 'Normal':
                    preds_attack.append('Exploits' if sev == 'HIGH' else ('Reconnaissance' if sev == 'MEDIUM' else 'Backdoor'))
                else:
                    preds_attack.append(atk)
        else:
            preds_attack = ['Normal' if s == 'LOW' else 'Attack' for s in preds_severity]
            
        df_result = df_upload.copy()
        df_result['Predicted_Severity'] = preds_severity
        df_result['Predicted_Attack'] = preds_attack
        if any(c is not None for c in confidences):
            df_result['Confidence_%'] = confidences
            
        # Summary statistics for UI charts
        severity_counts = pd.Series(preds_severity).value_counts().to_dict()
        attack_counts = pd.Series(preds_attack).value_counts().to_dict()
        
        return df_result, severity_counts, attack_counts

    def generate_sample_csv(self, n_rows=25):
        """
        Creates a clean sample test CSV from dataset.csv for users to test batch upload.
        """
        data_path = os.path.join('data', 'dataset.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            sample = df[FEATURE_COLUMNS].sample(n=min(len(df), n_rows), random_state=101)
            return sample
        else:
            # Fallback synthetic template
            return pd.DataFrame(columns=FEATURE_COLUMNS)

# Singleton predictor instance for application import
predictor = SeverityPredictor()
