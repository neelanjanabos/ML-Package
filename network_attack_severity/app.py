"""
Flask Web Application for Network Attack Severity Prediction System.
Provides web routes for:
- Home (Dashboard overview & summary metrics)
- Dataset (Dataset exploration, schema, class distributions, and data preview)
- Models (5 algorithm performance comparison, bar charts, confusion matrices)
- Predict (Single flow prediction with pre-filled sample presets and confidence score)
- Upload (Batch CSV prediction, table preview, chart, and CSV export)
- Results (Prediction history log and aggregate statistics from SQLite)
"""

import os
import io
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash

from preprocessing import (
    load_data, 
    FEATURE_COLUMNS, 
    FEATURE_INFO, 
    CATEGORICAL_FEATURES, 
    NUMERICAL_FEATURES, 
    SEVERITY_CLASSES,
    SEVERITY_MAPPING,
    get_sample_presets
)
from predict import predictor
from database import (
    init_db, 
    log_prediction, 
    get_prediction_history, 
    get_history_summary, 
    clear_prediction_history
)

app = Flask(__name__)
app.secret_key = 'network-severity-ml-student-project-secret-key'

# Initialize SQLite database
init_db()

# Load metrics.json cache if present
METRICS_PATH = os.path.join('models', 'metrics.json')
def get_metrics_data():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    return {}

# ------------------------------------------------------------------------------
# 1. HOME ROUTE
# ------------------------------------------------------------------------------
@app.route('/')
def home():
    metrics = get_metrics_data()
    dataset_records = metrics.get('total_records', 15000)
    features_count = metrics.get('features_count', len(FEATURE_COLUMNS))
    best_model = metrics.get('best_model', 'KNN')
    best_accuracy = metrics.get('best_accuracy', 85.39)
    attack_categories = metrics.get('attack_categories', {})
    severity_distribution = metrics.get('severity_distribution', {})
    
    return render_template(
        'index.html',
        dataset_records=dataset_records,
        features_count=features_count,
        attack_count=len(attack_categories),
        best_model=best_model,
        best_accuracy=best_accuracy,
        severity_distribution=severity_distribution,
        attack_categories=attack_categories,
        metrics=metrics
    )

# ------------------------------------------------------------------------------
# 2. DATASET ROUTE
# ------------------------------------------------------------------------------
@app.route('/dataset')
def dataset_page():
    data_path = os.path.join('data', 'dataset.csv')
    df = load_data(data_path)
    
    # Dataset statistics
    rows_count = len(df)
    cols_count = len(df.columns)
    
    # Distributions
    severity_counts = df['severity'].value_counts().to_dict()
    attack_counts = df['attack_cat'].value_counts().to_dict()
    binary_counts = df['label'].map({0: 'Normal (Benign)', 1: 'Attack Traffic'}).value_counts().to_dict()
    
    # Preview top 20 rows
    preview_cols = ['dur', 'proto', 'service', 'state', 'sbytes', 'dbytes', 'rate', 'smean', 'dmean', 'ct_src_dport_ltm', 'attack_cat', 'severity']
    preview_data = df[preview_cols].head(20).to_dict(orient='records')
    
    return render_template(
        'dataset.html',
        rows_count=rows_count,
        cols_count=cols_count,
        feature_info=FEATURE_INFO,
        feature_columns=FEATURE_COLUMNS,
        severity_mapping=SEVERITY_MAPPING,
        severity_counts=severity_counts,
        attack_counts=attack_counts,
        binary_counts=binary_counts,
        preview_data=preview_data
    )

# ------------------------------------------------------------------------------
# 3. MODEL COMPARISON ROUTE
# ------------------------------------------------------------------------------
@app.route('/models')
def models_page():
    metrics = get_metrics_data()
    models_dict = metrics.get('models', {})
    best_model = metrics.get('best_model', 'KNN')
    
    # Prepare comparison table list
    models_table = []
    for name, data in models_dict.items():
        models_table.append({
            'name': name,
            'accuracy': data.get('accuracy', 0),
            'precision': data.get('precision', 0),
            'recall': data.get('recall', 0),
            'f1_score': data.get('f1_score', 0),
            'train_time': data.get('train_time', 0),
            'is_best': (name == best_model)
        })
        
    # Sort table by F1 Score descending
    models_table.sort(key=lambda x: x['f1_score'], reverse=True)
    
    return render_template(
        'models.html',
        models_table=models_table,
        models_dict=models_dict,
        best_model=best_model,
        classes=SEVERITY_CLASSES
    )

# ------------------------------------------------------------------------------
# 4. PREDICTION ROUTE (Single Flow)
# ------------------------------------------------------------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    presets = get_sample_presets()
    result = None
    selected_model = 'Decision Tree'
    form_values = {}
    
    # Default initial form values
    for col in FEATURE_COLUMNS:
        form_values[col] = presets['normal']['values'].get(col, 0)
        
    if request.method == 'POST':
        selected_model = request.form.get('model_name', 'Decision Tree')
        for col in FEATURE_COLUMNS:
            form_values[col] = request.form.get(col, '')
            
        result = predictor.predict_single(form_values, model_name=selected_model)
        
        # Log to SQLite
        log_prediction(
            model_used=result['model_used'],
            severity=result['predicted_severity'],
            attack=result['predicted_attack'],
            confidence=result.get('confidence'),
            features=form_values
        )
        
    available_models = list(predictor.models.keys())
    
    return render_template(
        'predict.html',
        feature_columns=FEATURE_COLUMNS,
        feature_info=FEATURE_INFO,
        categorical_features=CATEGORICAL_FEATURES,
        numerical_features=NUMERICAL_FEATURES,
        available_models=available_models,
        selected_model=selected_model,
        form_values=form_values,
        presets=presets,
        result=result
    )

# ------------------------------------------------------------------------------
# 5. BATCH CSV UPLOAD & PREDICTION
# ------------------------------------------------------------------------------
# In-memory store for last batch prediction result to allow CSV export
LATEST_BATCH_RESULT = {}

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    global LATEST_BATCH_RESULT
    results_table = None
    severity_distribution = None
    attack_distribution = None
    total_processed = 0
    selected_model = 'Decision Tree'
    
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            flash('No file uploaded. Please select a CSV file.', 'danger')
            return redirect(request.url)
            
        file = request.files['csv_file']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)
            
        if not file.filename.endswith('.csv'):
            flash('Invalid file format. Please upload a .csv file.', 'danger')
            return redirect(request.url)
            
        selected_model = request.form.get('model_name', 'Decision Tree')
        
        try:
            df_upload = pd.read_csv(file)
            if df_upload.empty:
                flash('The uploaded CSV file is empty.', 'warning')
                return redirect(request.url)
                
            df_result, sev_dist, atk_dist = predictor.predict_batch(df_upload, model_name=selected_model)
            
            # Store in cache for download
            LATEST_BATCH_RESULT['df'] = df_result
            LATEST_BATCH_RESULT['model'] = selected_model
            
            total_processed = len(df_result)
            severity_distribution = sev_dist
            attack_distribution = atk_dist
            
            # Convert preview to records (first 50 rows)
            results_table = df_result.head(50).to_dict(orient='records')
            
            # Log summary batch to history
            for _, row in df_result.head(20).iterrows():
                log_prediction(
                    model_used=selected_model,
                    severity=row.get('Predicted_Severity', 'UNKNOWN'),
                    attack=row.get('Predicted_Attack', 'UNKNOWN'),
                    confidence=row.get('Confidence_%', 0.0),
                    features=row.to_dict()
                )
                
            flash(f'Successfully analyzed {total_processed} network flows using {selected_model}!', 'success')
            
        except Exception as e:
            flash(f'Error processing CSV file: {str(e)}', 'danger')
            
    available_models = list(predictor.models.keys())
    
    return render_template(
        'upload.html',
        available_models=available_models,
        selected_model=selected_model,
        results_table=results_table,
        severity_distribution=severity_distribution,
        attack_distribution=attack_distribution,
        total_processed=total_processed
    )

@app.route('/download-predictions')
def download_predictions():
    global LATEST_BATCH_RESULT
    if 'df' not in LATEST_BATCH_RESULT:
        flash('No prediction results available for download.', 'warning')
        return redirect(url_for('upload_page'))
        
    df_result = LATEST_BATCH_RESULT['df']
    csv_buffer = io.StringIO()
    df_result.to_csv(csv_buffer, index=False)
    csv_bytes = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
    
    return send_file(
        csv_bytes,
        mimetype='text/csv',
        as_attachment=True,
        download_name='network_attack_predictions.csv'
    )

@app.route('/download-sample-csv')
def download_sample_csv():
    sample_df = predictor.generate_sample_csv(n_rows=20)
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    csv_bytes = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
    
    return send_file(
        csv_bytes,
        mimetype='text/csv',
        as_attachment=True,
        download_name='unsw_nb15_sample_test_traffic.csv'
    )

# ------------------------------------------------------------------------------
# 6. RESULTS & HISTORY ROUTE
# ------------------------------------------------------------------------------
@app.route('/results')
def results_page():
    history = get_prediction_history(limit=100)
    summary = get_history_summary()
    metrics = get_metrics_data()
    best_model = metrics.get('best_model', 'KNN')
    
    return render_template(
        'results.html',
        history=history,
        total_predictions=summary['total'],
        severity_dist=summary['severity_dist'],
        attack_dist=summary['attack_dist'],
        model_dist=summary['model_dist'],
        best_model=best_model,
        models=metrics.get('models', {})
    )

@app.route('/clear-history', methods=['POST'])
def clear_history():
    clear_prediction_history()
    flash('Prediction history successfully cleared.', 'info')
    return redirect(url_for('results_page'))

# ------------------------------------------------------------------------------
# 7. JSON API ENDPOINTS FOR CLIENT-SIDE CHARTS
# ------------------------------------------------------------------------------
@app.route('/api/metrics')
def api_metrics():
    return jsonify(get_metrics_data())

@app.route('/api/presets')
def api_presets():
    return jsonify(get_sample_presets())

if __name__ == '__main__':
    print("\nStarting Network Attack Severity Prediction System Flask Server...")
    print("Accessible locally at: http://127.0.0.1:5000\n")
    app.run(host='127.0.0.1', port=5000, debug=True)
