"""
Data Preprocessing Module for Network Attack Severity Prediction System.
Handles dataset loading, cleaning, categorical encoding, feature scaling,
and CVSS-inspired severity mapping for the UNSW-NB15 network intrusion dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==============================================================================
# 1. SEVERITY MAPPING
# ==============================================================================
# Mapping UNSW-NB15 attack categories into 4 standard incident severity tiers:
# LOW:      Benign / normal network operations
# MEDIUM:   Reconnaissance, port scans, fuzzing, probing (early threat discovery)
# HIGH:     Active attacks, Denial of Service (DoS), known exploits (system impact)
# CRITICAL: Remote code execution, memory shellcode, persistent backdoors, worms
SEVERITY_MAPPING = {
    'Normal': 'LOW',
    'Reconnaissance': 'MEDIUM',
    'Analysis': 'MEDIUM',
    'Fuzzers': 'MEDIUM',
    'DoS': 'HIGH',
    'Generic': 'HIGH',
    'Exploits': 'HIGH',
    'Backdoor': 'CRITICAL',
    'Shellcode': 'CRITICAL',
    'Worms': 'CRITICAL'
}

# The 4 severity classes in order of increasing urgency
SEVERITY_CLASSES = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

# ==============================================================================
# 2. FEATURE DEFINITIONS
# ==============================================================================
CATEGORICAL_FEATURES = ['proto', 'service', 'state']
NUMERICAL_FEATURES = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 
    'rate', 'sload', 'dload', 'smean', 'dmean', 
    'ct_src_dport_ltm'
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Human-readable labels and descriptions for UI display and viva defense
FEATURE_INFO = {
    'dur': {'name': 'Duration', 'unit': 'seconds', 'desc': 'Total connection duration'},
    'proto': {'name': 'Protocol', 'unit': 'text', 'desc': 'Transaction protocol (tcp, udp, arp, etc.)'},
    'service': {'name': 'Service', 'unit': 'text', 'desc': 'Application service (http, ftp, dns, etc.)'},
    'state': {'name': 'State', 'unit': 'text', 'desc': 'Connection state (FIN, CON, INT, REQ, etc.)'},
    'spkts': {'name': 'Source Packets', 'unit': 'count', 'desc': 'Packets sent from source to destination'},
    'dpkts': {'name': 'Destination Packets', 'unit': 'count', 'desc': 'Packets sent from destination to source'},
    'sbytes': {'name': 'Source Bytes', 'unit': 'bytes', 'desc': 'Volume of bytes sent from source'},
    'dbytes': {'name': 'Destination Bytes', 'unit': 'bytes', 'desc': 'Volume of bytes sent from destination'},
    'rate': {'name': 'Flow Rate', 'unit': 'pkts/sec', 'desc': 'Packets per second in the flow'},
    'sload': {'name': 'Source Load', 'unit': 'bps', 'desc': 'Source bits per second'},
    'dload': {'name': 'Destination Load', 'unit': 'bps', 'desc': 'Destination bits per second'},
    'smean': {'name': 'Source Mean Size', 'unit': 'bytes', 'desc': 'Average size of packet transmitted by source'},
    'dmean': {'name': 'Destination Mean Size', 'unit': 'bytes', 'desc': 'Average size of packet transmitted by destination'},
    'ct_src_dport_ltm': {'name': 'Conn to Dest Port', 'unit': 'count', 'desc': 'Number of connections to the same destination port'}
}

def load_data(filepath='data/dataset.csv'):
    """
    Load dataset from CSV, clean column names, strip strings, and ensure severity column exists.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Ensure severity column is properly populated
    if 'severity' not in df.columns or df['severity'].isnull().any():
        if 'attack_cat' in df.columns:
            df['severity'] = df['attack_cat'].map(SEVERITY_MAPPING).fillna('LOW')
            
    # Handle any unexpected missing values
    for col in NUMERICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
            
    return df

def preprocess_features(df, encoders=None, scaler=None, is_training=True):
    """
    Preprocess features for training or inference:
    - Encodes categorical variables (proto, service, state)
    - Scales numerical variables using StandardScaler
    - Avoids data leakage: fits scaler and encoders only if is_training=True
    """
    df_clean = df.copy()
    
    # Categorical encoding
    if is_training:
        encoders = {}
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            # Fit on observed values, plus 'other' for robust unseen category handling
            unique_vals = list(df_clean[col].astype(str).unique())
            if 'other' not in unique_vals:
                unique_vals.append('other')
            le.fit(unique_vals)
            encoders[col] = le
            # Transform
            df_clean[col] = df_clean[col].astype(str).map(
                lambda s: le.transform([s])[0] if s in le.classes_ else le.transform(['other'])[0]
            )
    else:
        if encoders is None:
            raise ValueError("Encoders must be provided when is_training=False")
        for col in CATEGORICAL_FEATURES:
            le = encoders[col]
            df_clean[col] = df_clean[col].astype(str).map(
                lambda s: le.transform([s])[0] if s in le.classes_ else le.transform(['other'])[0]
            )

    # Feature matrix X
    X = df_clean[FEATURE_COLUMNS].copy()

    # Numerical scaling
    if is_training:
        scaler = StandardScaler()
        X[NUMERICAL_FEATURES] = scaler.fit_transform(X[NUMERICAL_FEATURES])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when is_training=False")
        X[NUMERICAL_FEATURES] = scaler.transform(X[NUMERICAL_FEATURES])

    return X, encoders, scaler

def get_sample_presets():
    """
    Predefined realistic packet traffic samples extracted from genuine UNSW-NB15 records.
    Helps professors, evaluators, and students verify each severity tier immediately.
    """
    return {
        'normal': {
            'name': 'Benign Traffic (Normal)',
            'description': 'Legitimate TCP network session with balanced bidirectional transmission and normal FIN handshake.',
            'expected_attack': 'Normal',
            'expected_severity': 'LOW',
            'values': {
                'dur': 0.7925, 'proto': 'tcp', 'service': '-', 'state': 'FIN',
                'spkts': 10, 'dpkts': 10, 'sbytes': 608, 'dbytes': 646,
                'rate': 23.97, 'sload': 5531.68, 'dload': 5874.89,
                'smean': 61, 'dmean': 65, 'ct_src_dport_ltm': 1
            }
        },
        'probe': {
            'name': 'Port Scanning / Probe (Reconnaissance)',
            'description': 'Targeted port scan probing service availability across destination ports.',
            'expected_attack': 'Reconnaissance',
            'expected_severity': 'MEDIUM',
            'values': {
                'dur': 0.7923, 'proto': 'tcp', 'service': '-', 'state': 'FIN',
                'spkts': 10, 'dpkts': 8, 'sbytes': 564, 'dbytes': 354,
                'rate': 21.46, 'sload': 5129.52, 'dload': 3130.22,
                'smean': 56, 'dmean': 44, 'ct_src_dport_ltm': 1
            }
        },
        'dos': {
            'name': 'Denial of Service Flood (DoS)',
            'description': 'High frequency packet burst saturating target bandwidth and destination ports.',
            'expected_attack': 'DoS',
            'expected_severity': 'HIGH',
            'values': {
                'dur': 0.00001, 'proto': 'rsvp', 'service': '-', 'state': 'INT',
                'spkts': 2, 'dpkts': 0, 'sbytes': 200, 'dbytes': 0,
                'rate': 111111.11, 'sload': 88888890.0, 'dload': 0.0,
                'smean': 100, 'dmean': 0, 'ct_src_dport_ltm': 4
            }
        },
        'critical': {
            'name': 'Malicious Shellcode / Payload Injection',
            'description': 'Injected binary shellcode payload attempting memory corruption and unauthorized execution.',
            'expected_attack': 'Shellcode',
            'expected_severity': 'CRITICAL',
            'values': {
                'dur': 0.00001, 'proto': 'udp', 'service': '-', 'state': 'INT',
                'spkts': 2, 'dpkts': 0, 'sbytes': 246, 'dbytes': 0,
                'rate': 125000.0, 'sload': 123000000.0, 'dload': 0.0,
                'smean': 123, 'dmean': 0, 'ct_src_dport_ltm': 1
            }
        }
    }
