# ============================================================================
# UPI FRAUD DETECTION - FLASK API SERVER
# ============================================================================
# This Flask API loads the trained ensemble model and provides real-time
# fraud scoring via HTTP endpoints.
#
# Run: python api_server.py
# Then access: http://localhost:8000/
#
# Requirements: pip install flask flask-cors gunicorn
# ============================================================================

import pickle
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ============================================================================
# LOAD MODEL ARTIFACTS
# ============================================================================

print("\n" + "="*80)
print("🔄 LOADING MODEL ARTIFACTS")
print("="*80)

try:
    # Load trained models and scaler
    with open('fraud_detection_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    xgb_model = artifacts['xgb_model']
    rf_model = artifacts['rf_model']
    nn_model = artifacts['nn_model']
    scaler = artifacts['scaler']
    feature_names = artifacts['feature_names']
    weights = artifacts['weights']
    metrics = artifacts['metrics']
    
    w_xgb = weights['xgb']
    w_rf = weights['rf']
    w_nn = weights['nn']
    
    # Load metadata
    with open('model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    print("✓ Models loaded successfully")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Ensemble AUC-ROC: {metrics['ensemble_auc']:.4f}")
    
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("   Run 'python train_model.py' first to generate model files")
    exit(1)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_feature_vector(txn_dict):
    """
    Convert a transaction dictionary into the exact feature vector
    expected by the trained model.
    """
    # Create empty feature dict with all expected features
    feature_dict = {col: 0 for col in feature_names}
    
    # Map input fields to features
    direct_fields = [
        'hour', 'amount', 'state_mismatch', 'account_age_days',
        'prev_transaction_count', 'is_new_beneficiary', 'device_new',
        'late_night_txn', 'weekend_txn', 'receiver_risk_score',
        'duplicate_txn_24h', 'txn_velocity', 'amount_over_avg',
        'day_of_week', 'day_of_month', 'month', 'transaction_type_encoded'
    ]
    
    for field in direct_fields:
        if field in txn_dict:
            feature_dict[field] = txn_dict[field]
    
    # Handle behavioral aggregates with defaults
    feature_dict['device_fraud_rate_x'] = txn_dict.get('device_fraud_rate', 0.0)
    feature_dict['device_fraud_rate_y'] = txn_dict.get('device_fraud_rate', 0.0)
    feature_dict['txn_in_last_hour'] = txn_dict.get('txn_in_last_hour', 1)
    feature_dict['typical_txn_count_x'] = txn_dict.get('typical_txn_count', 10)
    feature_dict['typical_txn_count_y'] = txn_dict.get('typical_txn_count', 10)
    feature_dict['state_pair_fraud_rate_x'] = txn_dict.get('state_pair_fraud_rate', 0.05)
    feature_dict['state_pair_fraud_rate_y'] = txn_dict.get('state_pair_fraud_rate', 0.05)
    feature_dict['receiver_fraud_rate_x'] = txn_dict.get('receiver_fraud_rate', 0.02)
    feature_dict['receiver_fraud_rate_y'] = txn_dict.get('receiver_fraud_rate', 0.02)
    
    # Create DataFrame in correct column order
    X = pd.DataFrame([feature_dict])[feature_names]
    X_scaled = scaler.transform(X)
    
    return X_scaled

def score_transaction(txn_dict):
    """
    Score a transaction using the ensemble model.
    """
    try:
        X_scaled = build_feature_vector(txn_dict)
        
        # Get predictions from each model
        p_xgb = xgb_model.predict_proba(X_scaled)[:, 1][0]
        p_rf = rf_model.predict_proba(X_scaled)[:, 1][0]
        p_nn = nn_model.predict_proba(X_scaled)[:, 1][0]
        
        # Weighted ensemble
        fraud_score = float(w_xgb * p_xgb + w_rf * p_rf + w_nn * p_nn)
        
        # Decision thresholds
        if fraud_score < 0.2:
            risk_level = "LOW"
            action = "ALLOW"
        elif fraud_score < 0.5:
            risk_level = "MEDIUM"
            action = "STEP_UP_AUTH"
        else:
            risk_level = "HIGH"
            action = "BLOCK_OR_REVIEW"
        
        # Individual model scores for transparency
        individual_scores = {
            'gradient_boosting': float(p_xgb),
            'random_forest': float(p_rf),
            'neural_network': float(p_nn)
        }
        
        return {
            'fraud_score': fraud_score,
            'risk_level': risk_level,
            'recommended_action': action,
            'individual_scores': individual_scores,
            'weights': weights,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'UPI Fraud Detection Ensemble',
        'version': '1.0.0',
        'ensemble_auc': metrics['ensemble_auc']
    }), 200

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get model metadata and performance information"""
    return jsonify(model_metadata), 200

@app.route('/score_transaction', methods=['POST'])
def score():
    """
    Main scoring endpoint.
    
    Expected JSON payload:
    {
        "amount": 1000,
        "hour": 14,
        "day_of_week": 3,
        "is_new_beneficiary": 0,
        "device_new": 0,
        "state_mismatch": 0,
        "account_age_days": 365,
        "prev_transaction_count": 100,
        "receiver_risk_score": 2,
        "duplicate_txn_24h": 0,
        "txn_velocity": 1,
        "transaction_type_encoded": 0,
        "device_fraud_rate": 0.01,
        "day_of_month": 15,
        "month": 12,
        "late_night_txn": 0,
        "weekend_txn": 0,
        "amount_over_avg": 0
    }
    """
    txn = request.get_json()
    
    if not txn:
        return jsonify({'error': 'No JSON payload provided'}), 400
    
    try:
        result = score_transaction(txn)
        if 'error' in result:
            return jsonify(result), 400
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_score', methods=['POST'])
def batch_score():
    """
    Score multiple transactions at once.
    
    Expected JSON:
    {
        "transactions": [
            {...},
            {...}
        ]
    }
    """
    data = request.get_json()
    
    if not data or 'transactions' not in data:
        return jsonify({'error': 'Expected JSON with transactions array'}), 400
    
    try:
        results = []
        for txn in data['transactions']:
            result = score_transaction(txn)
            results.append(result)
        
        return jsonify({'results': results, 'count': len(results)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Serve the HTML interface"""
    # For production, load from index.html file
    # For demo, we can return a simple message
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>UPI Fraud Detection API</title>
        <style>
            body { font-family: system-ui; margin: 40px; max-width: 800px; }
            h1 { color: #2080B6; }
            code { background: #f0f0f0; padding: 10px; display: block; margin: 10px 0; }
            .endpoint { margin: 20px 0; padding: 15px; border-left: 4px solid #2080B6; }
        </style>
    </head>
    <body>
        <h1>🛡️ UPI Fraud Detection API</h1>
        <p>Ensemble ML-powered real-time transaction fraud risk assessment</p>
        
        <div class="endpoint">
            <h3>GET /health</h3>
            <p>Health check and model status</p>
        </div>
        
        <div class="endpoint">
            <h3>GET /model_info</h3>
            <p>Get detailed model metadata and performance metrics</p>
        </div>
        
        <div class="endpoint">
            <h3>POST /score_transaction</h3>
            <p>Score a single transaction</p>
            <code>
{
    "amount": 1000,
    "hour": 14,
    "day_of_week": 3,
    "is_new_beneficiary": 0,
    "device_new": 0,
    "state_mismatch": 0,
    "account_age_days": 365,
    "prev_transaction_count": 100,
    "receiver_risk_score": 2,
    "duplicate_txn_24h": 0,
    "txn_velocity": 1,
    "transaction_type_encoded": 0,
    "device_fraud_rate": 0.01
}
            </code>
        </div>
        
        <div class="endpoint">
            <h3>POST /batch_score</h3>
            <p>Score multiple transactions at once</p>
        </div>
        
        <hr>
        <p><strong>To use the web interface:</strong> Load index.html in your browser</p>
        <p><strong>Documentation:</strong> See README.md</p>
    </body>
    </html>
    ''', 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 STARTING UPI FRAUD DETECTION API SERVER")
    print("="*80)
    print("\n📍 Server running on: http://localhost:8000")
    print("📡 API Endpoints:")
    print("   • GET  /health")
    print("   • GET  /model_info")
    print("   • POST /score_transaction")
    print("   • POST /batch_score")
    print("   • GET  /")
    print("\n🌐 Open index.html in your browser to use the web interface\n")
    
    # For production use Gunicorn:
    # gunicorn -w 4 -b 0.0.0.0:8000 api_server:app
    
    app.run(host='0.0.0.0', port=8000, debug=True)
