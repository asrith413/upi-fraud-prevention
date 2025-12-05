# ============================================================================
# UPI FRAUD DETECTION - MODEL TRAINING SCRIPT
# ============================================================================
# This script generates synthetic UPI transaction data, trains an ensemble
# ML model (XGBoost + Random Forest + Neural Network), and saves artifacts
# for deployment.
#
# Run: python train_model.py
# ============================================================================

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, f1_score, 
                             precision_recall_curve, auc, accuracy_score, 
                             precision_score, recall_score)
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("🚀 UPI FRAUD DETECTION SYSTEM - MODEL TRAINING")
print("="*80)

# ============================================================================
# PART 1: SYNTHETIC DATA GENERATION
# ============================================================================

np.random.seed(42)
random.seed(42)

def generate_upi_dataset(n_samples=10000, fraud_ratio=0.15):
    """
    Generate realistic UPI transaction dataset with authentic fraud patterns.
    
    Fraud types included:
    - Social engineering (impersonation)
    - Fake refund scams
    - QR code swapping at merchants
    - Verification transactions (test transfers)
    - Account takeover
    """
    
    data = []
    
    # Indian states
    indian_states = ['Delhi', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'West Bengal',
                     'Gujarat', 'Rajasthan', 'Punjab', 'Telangana', 'Uttar Pradesh']
    
    # Fraud pattern distribution
    fraud_patterns = {
        'social_engineering': 0.30,
        'fake_refund': 0.25,
        'qr_swapping': 0.20,
        'verification_txn': 0.15,
        'account_takeover': 0.10
    }
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud
    
    print(f"\n📊 Generating {n_samples} transactions ({fraud_ratio*100:.0f}% fraud)")
    
    # LEGITIMATE TRANSACTIONS
    for i in range(n_legitimate):
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365))
        hour = random.choices(range(24), 
                            weights=[2,1,1,1,2,3,4,5,6,7,6,5,4,5,6,7,8,7,6,5,4,3,2,1])[0]
        
        # Log-normal distribution for amount (realistic UPI patterns)
        amount = np.random.lognormal(6, 1.5)
        amount = min(max(amount, 100), 50000)
        
        # Device consistency (users have 1-2 devices)
        device_id = f"DEV_{i % 500}"
        sender_state = random.choice(indian_states)
        receiver_state = random.choice(indian_states)
        
        state_mismatch = 1 if sender_state != receiver_state else 0
        txn_type = random.choices(['P2P', 'Merchant'], weights=[0.6, 0.4])[0]
        
        account_age = random.randint(30, 1825)
        prev_transactions = random.randint(10, 1000)
        
        data.append({
            'transaction_id': f"TXN_{i:06d}",
            'timestamp': timestamp,
            'hour': hour,
            'amount': amount,
            'device_id': device_id,
            'sender_state': sender_state,
            'receiver_state': receiver_state,
            'state_mismatch': state_mismatch,
            'transaction_type': txn_type,
            'account_age_days': account_age,
            'prev_transaction_count': prev_transactions,
            'is_new_beneficiary': random.choices([0, 1], weights=[0.9, 0.1])[0],
            'device_new': random.choices([0, 1], weights=[0.95, 0.05])[0],
            'late_night_txn': 1 if hour in [22, 23, 0, 1, 2, 3, 4, 5] else 0,
            'weekend_txn': 1 if timestamp.weekday() >= 5 else 0,
            'receiver_risk_score': random.randint(0, 5),
            'duplicate_txn_24h': random.randint(0, 2),
            'txn_velocity': random.randint(1, 10),
            'amount_over_avg': 0 if amount <= 5000 else 1,
            'fraud': 0
        })
    
    # FRAUDULENT TRANSACTIONS
    fraud_types = list(fraud_patterns.keys())
    fraud_weights = list(fraud_patterns.values())
    
    for i in range(n_fraud):
        timestamp = datetime.now() - timedelta(days=random.randint(0, 90))
        
        fraud_type = np.random.choice(fraud_types, p=np.array(fraud_weights)/sum(fraud_weights))
        
        # Fraud-specific patterns
        if fraud_type == 'social_engineering':
            amount = random.uniform(500, 15000)
            hour = random.choices(range(24), 
                                weights=[1,1,1,1,1,2,3,4,5,6,5,4,4,5,6,7,7,7,6,5,4,3,2,2])[0]
            is_new_benef = 1
            device_new = random.choices([0, 1], weights=[0.4, 0.6])[0]
            late_night = random.choices([0, 1], weights=[0.7, 0.3])[0]
            state_mismatch = 1
            
        elif fraud_type == 'fake_refund':
            amount = random.uniform(500, 5000)
            hour = random.choices(range(24), 
                                weights=[1,1,1,1,2,3,4,5,6,7,6,5,4,5,6,7,7,6,6,5,5,4,3,2])[0]
            is_new_benef = 1
            device_new = 1
            late_night = 0
            state_mismatch = 1
            
        elif fraud_type == 'qr_swapping':
            amount = random.uniform(100, 2000)
            hour = random.choices([10,11,12,13,14,15,16,17,18,19,20], 
                                weights=[1,1,2,2,2,2,2,2,2,2,1])[0]
            is_new_benef = 1
            device_new = 0
            late_night = 0
            state_mismatch = 1
            
        elif fraud_type == 'verification_txn':
            amount = random.uniform(1, 100)
            hour = random.randint(0, 23)
            is_new_benef = 1
            device_new = random.choices([0, 1], weights=[0.5, 0.5])[0]
            late_night = random.choices([0, 1], weights=[0.7, 0.3])[0]
            state_mismatch = 1
            
        else:  # account_takeover
            amount = random.uniform(5000, 50000)
            hour = random.choices(range(24), 
                                weights=[3,2,2,1,1,1,2,3,4,5,6,7,7,6,5,4,3,3,4,5,6,7,7,6])[0]
            is_new_benef = random.choices([0, 1], weights=[0.5, 0.5])[0]
            device_new = 1
            late_night = 1
            state_mismatch = 1
        
        data.append({
            'transaction_id': f"TXN_FRAUD_{i:06d}",
            'timestamp': timestamp,
            'hour': hour,
            'amount': amount,
            'device_id': f"FDEV_{random.randint(0, 1000)}",
            'sender_state': random.choice(indian_states),
            'receiver_state': random.choice(indian_states),
            'state_mismatch': state_mismatch,
            'transaction_type': random.choices(['P2P', 'Merchant'], weights=[0.7, 0.3])[0],
            'account_age_days': random.randint(5, 100),
            'prev_transaction_count': random.randint(1, 50),
            'is_new_beneficiary': is_new_benef,
            'device_new': device_new,
            'late_night_txn': late_night,
            'weekend_txn': random.choices([0, 1], weights=[0.7, 0.3])[0],
            'receiver_risk_score': random.randint(6, 10),
            'duplicate_txn_24h': random.randint(1, 5),
            'txn_velocity': random.randint(3, 20),
            'amount_over_avg': 1 if amount > 5000 else 0,
            'fraud': 1
        })
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Generate dataset
df = generate_upi_dataset(n_samples=10000, fraud_ratio=0.15)

print(f"✓ Dataset created: {len(df)} transactions")
print(f"✓ Fraud cases: {df['fraud'].sum()} ({df['fraud'].sum()/len(df)*100:.1f}%)")
print(f"✓ Legitimate cases: {(df['fraud']==0).sum()} ({(df['fraud']==0).sum()/len(df)*100:.1f}%)")

# ============================================================================
# PART 2: FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("🔧 FEATURE ENGINEERING")
print("="*80)

# Extract date features
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_month'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# Encode categorical variables
df['transaction_type_encoded'] = (df['transaction_type'] == 'Merchant').astype(int)

# User behavior metrics
user_fraud_rate = df.groupby('device_id')['fraud'].mean().reset_index()
user_fraud_rate.columns = ['device_id', 'device_fraud_rate']
df = df.merge(user_fraud_rate, on='device_id', how='left')

# Transaction velocity
df['txn_in_last_hour'] = df['txn_velocity']
txn_by_hour = df.groupby(['sender_state', 'hour']).size().reset_index(name='typical_txn_count')
df = df.merge(txn_by_hour, on=['sender_state', 'hour'], how='left', validate='m:1')

# State pair fraud rate
state_pair_fraud = df.groupby(['sender_state', 'receiver_state'])['fraud'].agg(['sum', 'count']).reset_index()
state_pair_fraud.columns = ['sender_state', 'receiver_state', 'fraud_count', 'total_count']
state_pair_fraud['state_pair_fraud_rate'] = state_pair_fraud['fraud_count'] / state_pair_fraud['total_count']
df = df.merge(state_pair_fraud[['sender_state', 'receiver_state', 'state_pair_fraud_rate']], 
              on=['sender_state', 'receiver_state'], how='left', validate='m:1')

# Beneficiary risk analysis
benef_fraud = df.groupby(['receiver_state', 'transaction_type'])['fraud'].mean().reset_index()
benef_fraud.columns = ['receiver_state', 'transaction_type', 'receiver_fraud_rate']
df = df.merge(benef_fraud, on=['receiver_state', 'transaction_type'], how='left', validate='m:1')

# Prepare model data
df_model = df.drop(['transaction_id', 'timestamp', 'sender_state', 'receiver_state', 
                    'device_id', 'transaction_type'], axis=1)
df_model = df_model.fillna(0)

X = df_model.drop('fraud', axis=1)
y = df_model['fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features engineered: {len(X.columns)}")
print(f"✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# ============================================================================
# PART 3: MODEL TRAINING - ENSEMBLE
# ============================================================================

print("\n" + "="*80)
print("🤖 TRAINING ENSEMBLE MODELS")
print("="*80)

# Model 1: Gradient Boosting
print("\n[1/3] Training Gradient Boosting...")
xgb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42,
    verbose=0
)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
print(f"   ✓ AUC-ROC: {xgb_auc:.4f}")

# Model 2: Random Forest
print("[2/3] Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred_proba)
print(f"   ✓ AUC-ROC: {rf_auc:.4f}")

# Model 3: Neural Network
print("[3/3] Training Neural Network...")
nn_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=500,
    learning_rate_init=0.001,
    alpha=0.001,
    random_state=42,
    verbose=0
)
nn_model.fit(X_train_scaled, y_train)
nn_pred = nn_model.predict(X_test_scaled)
nn_pred_proba = nn_model.predict_proba(X_test_scaled)[:, 1]
nn_auc = roc_auc_score(y_test, nn_pred_proba)
print(f"   ✓ AUC-ROC: {nn_auc:.4f}")

# ============================================================================
# PART 4: ENSEMBLE VOTING
# ============================================================================

print("\n" + "="*80)
print("⚖️ ENSEMBLE VOTING")
print("="*80)

total_auc = xgb_auc + rf_auc + nn_auc
w_xgb = xgb_auc / total_auc
w_rf = rf_auc / total_auc
w_nn = nn_auc / total_auc

ensemble_pred_proba = (w_xgb * xgb_pred_proba + w_rf * rf_pred_proba + w_nn * nn_pred_proba)
ensemble_pred = (ensemble_pred_proba >= 0.5).astype(int)
ensemble_auc = roc_auc_score(y_test, ensemble_pred_proba)

print(f"\nWeights assigned:")
print(f"  • Gradient Boosting: {w_xgb:.3f}")
print(f"  • Random Forest: {w_rf:.3f}")
print(f"  • Neural Network: {w_nn:.3f}")
print(f"\n✓ Ensemble AUC-ROC: {ensemble_auc:.4f}")

# ============================================================================
# PART 5: MODEL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("📈 MODEL EVALUATION")
print("="*80)

metrics = {}

for name, pred, pred_proba in [
    ('Gradient Boosting', xgb_pred, xgb_pred_proba),
    ('Random Forest', rf_pred, rf_pred_proba),
    ('Neural Network', nn_pred, nn_pred_proba),
    ('Ensemble', ensemble_pred, ensemble_pred_proba)
]:
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred_proba)
    
    metrics[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

# Detailed evaluation
print("\n" + "="*80)
print("ENSEMBLE MODEL - DETAILED REPORT")
print("="*80)
print("\n" + classification_report(y_test, ensemble_pred, target_names=['Legitimate', 'Fraud']))

cm = confusion_matrix(y_test, ensemble_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}")
print(f"  True Positives:  {cm[1,1]}")

fpr = cm[0,1] / (cm[0,1] + cm[0,0])
fnr = cm[1,0] / (cm[1,0] + cm[1,1])

print(f"\nFalse Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
print(f"False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")

# Feature importance
print("\n" + "="*80)
print("TOP 10 IMPORTANT FEATURES")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n")
for idx, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"{idx:2d}. {row['feature']:35s} {row['importance']:.4f} ({row['importance']*100:.2f}%)")

# ============================================================================
# PART 6: SAVE ARTIFACTS
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING MODEL ARTIFACTS")
print("="*80)

model_artifacts = {
    'xgb_model': xgb_model,
    'rf_model': rf_model,
    'nn_model': nn_model,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'weights': {
        'xgb': float(w_xgb),
        'rf': float(w_rf),
        'nn': float(w_nn)
    },
    'metrics': {
        'xgb_auc': float(xgb_auc),
        'rf_auc': float(rf_auc),
        'nn_auc': float(nn_auc),
        'ensemble_auc': float(ensemble_auc),
        'recall': float(metrics['Ensemble']['recall']),
        'precision': float(metrics['Ensemble']['precision']),
        'f1': float(metrics['Ensemble']['f1']),
        'fpr': float(fpr),
        'fnr': float(fnr)
    }
}

with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)
print("✓ Saved: fraud_detection_model.pkl")

model_metadata = {
    'model_name': 'UPI Fraud Detection System - Ensemble ML',
    'version': '1.0.0',
    'trained_date': datetime.now().isoformat(),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'features_count': len(X.columns),
    'fraud_cases_in_training': int(y_train.sum()),
    'performance': {
        'ensemble_auc_roc': float(ensemble_auc),
        'accuracy': float(metrics['Ensemble']['accuracy']),
        'precision': float(metrics['Ensemble']['precision']),
        'recall': float(metrics['Ensemble']['recall']),
        'f1_score': float(metrics['Ensemble']['f1']),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr)
    },
    'model_components': {
        'gradient_boosting': {'auc': float(xgb_auc), 'weight': float(w_xgb)},
        'random_forest': {'auc': float(rf_auc), 'weight': float(w_rf)},
        'neural_network': {'auc': float(nn_auc), 'weight': float(w_nn)}
    },
    'top_features': feature_importance.head(10)[['feature', 'importance']].to_dict('records')
}

with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)
print("✓ Saved: model_metadata.json")

print("\n" + "="*80)
print("✅ MODEL TRAINING COMPLETE - READY FOR DEPLOYMENT")
print("="*80)
print("\nNext steps:")
print("  1. Run: python api_server.py")
print("  2. Open: index.html in your browser")
print("  3. Start scoring transactions!\n")
