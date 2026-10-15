#!/usr/bin/env python3
"""
SHIELD: Asymmetric Edge/Cloud Intrusion Detection Benchmark (NSL-KDD)
- Edge Tier: Lightweight Linear Classifier (fast on-premise resolution)
- Cloud Tier: High-Capacity Polynomial Classifier (degree-2) evaluated under CKKS FHE
"""

import time
import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from Pyfhel import Pyfhel

# ---------------------------------------------------------
# 1. Download & Prepare Open Cybersecurity Dataset (NSL-KDD)
# ---------------------------------------------------------
print("[1] Fetching NSL-KDD Intrusion Detection Dataset...")
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv"
local_file = "/workspace/nsl_kdd_train.csv"

if not os.path.exists(local_file):
    try:
        urllib.request.urlretrieve(url, local_file)
        print("    Dataset downloaded successfully.")
    except Exception as e:
        print(f"    Download failed: {e}. Check internet access.")
        exit(1)
else:
    print("    Using cached dataset in /workspace/nsl_kdd_train.csv.")

# NSL-KDD feature definitions
col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

df = pd.read_csv(local_file, names=col_names)

# Binary target: 0 = normal traffic, 1 = attack
df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Top 10 continuous traffic features
selected_features = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    "same_srv_rate", "diff_srv_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate"
]

X_raw = df[selected_features].values
y_raw = df['binary_label'].values

# Standardize inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Subsample: 2,500 train samples, 500 test samples
np.random.seed(42)
indices = np.random.permutation(len(X_scaled))
train_idx, test_idx = indices[:2500], indices[2500:3000]

X_train, y_train = X_scaled[train_idx], y_raw[train_idx]
X_test, y_test = X_scaled[test_idx], y_raw[test_idx]

print(f"    Train size: {len(X_train)} flows | Evaluation size: {len(X_test)} flows.")

# ---------------------------------------------------------
# 2. Train Asymmetric Models (Lightweight Edge vs Deep Cloud)
# ---------------------------------------------------------
print("[2] Training Asymmetric Model Tiers...")

# Tier A: Lightweight Edge Linear Classifier (10 inputs)
edge_clf = LogisticRegression(max_iter=500, random_state=42)
edge_clf.fit(X_train, y_train)

# Tier B: High-Capacity Polynomial Interaction Model for Cloud
# Computes non-linear feature interactions (x_i * x_j), yielding 65 features
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_transformer.fit_transform(X_train)
X_test_poly = poly_transformer.transform(X_test)

cloud_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
cloud_clf.fit(X_train_poly, y_train)

cloud_weights = cloud_clf.coef_[0].astype(np.float64)
cloud_intercept = float(cloud_clf.intercept_[0])
n_poly_features = len(cloud_weights)

print(f"    Edge Model Parameters : {X_train.shape[1]} linear weights")
print(f"    Cloud Model Parameters: {n_poly_features} polynomial weights (x_i * x_j interactions)")

# ---------------------------------------------------------
# 3. Cryptographic Setup (CKKS Context on DGX Spark)
# ---------------------------------------------------------
print("[3] Initializing Pyfhel CKKS Homomorphic Encryption Context...")
HE = Pyfhel()
HE.contextGen(scheme='CKKS', n=8192, scale=2**30, qi_sizes=[60, 30, 30, 60])
HE.keyGen()
HE.relinKeyGen()

# ---------------------------------------------------------
# 4. Asymmetric Hybrid Pipeline Evaluation
# ---------------------------------------------------------
GAMMA = 0.80  # Uncertainty gating threshold

edge_preds, edge_trues = [], []
cloud_preds, cloud_trues = [], []
all_preds, all_trues = [], []

print(f"[4] Running Asymmetric Hybrid Inference (Confidence Threshold gamma = {GAMMA})...")

for sample_linear, sample_poly, true_label in zip(X_test, X_test_poly, y_test):
    # Edge evaluation
    probs = edge_clf.predict_proba([sample_linear])[0]
    confidence = max(probs)
    
    if confidence >= GAMMA:
        # High confidence: resolve locally at the edge
        pred = int(np.argmax(probs))
        edge_preds.append(pred)
        edge_trues.append(true_label)
    else:
        # Ambiguous flow: escalate to Cloud with high-capacity polynomial evaluation under FHE
        ctxt_x_poly = HE.encrypt(sample_poly.astype(np.float64))
        ctxt_w_poly = HE.encrypt(cloud_weights)
        
        # Homomorphic dot product over non-linear feature space
        ctxt_res = ctxt_x_poly * ctxt_w_poly
        
        decrypted_vector = HE.decrypt(ctxt_res)[:n_poly_features]
        score = float(np.sum(decrypted_vector) + cloud_intercept)
        pred = 1 if score > 0 else 0
        
        cloud_preds.append(pred)
        cloud_trues.append(true_label)
        
    all_preds.append(pred)
    all_trues.append(true_label)

# ---------------------------------------------------------
# 5. Summary Performance Metrics
# ---------------------------------------------------------
total = len(X_test)
n_edge = len(edge_preds)
n_cloud = len(cloud_preds)

print("\n" + "=" * 65)
print("     SHIELD: ASYMMETRIC REAL ATTACK EVALUATION (NSL-KDD)")
print("=" * 65)
print(f"Total Flows Evaluated         : {total}")
print(f"Resolved Locally at Edge (Fast): {n_edge} ({n_edge / total * 100:.1f}%)")
print(f"Escalated to Encrypted Cloud  : {n_cloud} ({n_cloud / total * 100:.1f}%)")
print("-" * 65)
print("DETECTION PERFORMANCE:")
print(f"  Overall Hybrid Pipeline Accuracy: {accuracy_score(all_trues, all_preds) * 100:.2f}%")
print(f"  Overall Hybrid Pipeline F1-Score: {f1_score(all_trues, all_preds):.4f}")
print("-" * 65)
if n_edge > 0:
    print(f"  Edge Tier Accuracy (Easy Flows) : {accuracy_score(edge_trues, edge_preds) * 100:.2f}%")
if n_cloud > 0:
    print(f"  Cloud FHE Accuracy (Hard Flows) : {accuracy_score(cloud_trues, cloud_preds) * 100:.2f}%")
print("=" * 65)