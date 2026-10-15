#!/usr/bin/env python3
"""
SHIELD: Hybrid Edge/Cloud Privacy-Preserving Intrusion Detection
End-to-End Feasibility Prototype using Pyfhel (CKKS) on ARM64 / DGX Spark
"""

import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from Pyfhel import Pyfhel

# ---------------------------------------------------------
# 1. Dataset Generation & Offline Plaintext Training
# ---------------------------------------------------------
print("[1] Generating synthetic tabular network flow dataset...")
# Simulating 1,000 flows with 10 continuous traffic metrics (rates, byte counts, durations)
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    class_sep=1.2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train baseline edge and cloud linear classifiers offline
edge_model = LogisticRegression(random_state=42)
edge_model.fit(X_train, y_train)

# Extract learned model parameters (weights W and bias b)
cloud_weights = edge_model.coef_[0].astype(np.float64)
cloud_intercept = float(edge_model.intercept_[0])

# ---------------------------------------------------------
# 2. Cryptographic Context Initialization (CKKS Scheme)
# ---------------------------------------------------------
print("[2] Initializing CKKS Homomorphic Encryption Context (Pyfhel)...")
HE = Pyfhel()
# n=8192 provides 4096 SIMD vector slots per ciphertext
HE.contextGen(scheme='CKKS', n=8192, scale=2**30, qi_sizes=[60, 30, 30, 60])
HE.keyGen()
HE.relinKeyGen()

# ---------------------------------------------------------
# 3. Hybrid Inference Simulation with Uncertainty Gating
# ---------------------------------------------------------
# Confidence threshold: flows with max class probability below GAMMA are offloaded
GAMMA = 0.75

edge_latencies = []
cloud_latencies = []

# Metric tracking lists
all_preds = []
all_trues = []
edge_preds = []
edge_trues = []
cloud_preds = []
cloud_trues = []
ckks_numerical_diffs = []

print(f"[3] Evaluating test flows (Confidence Gating Threshold: gamma = {GAMMA})...")

for sample, true_label in zip(X_test, y_test):
    # --- Edge Tier: Plaintext Evaluation ---
    t0_edge = time.perf_counter()
    probs = edge_model.predict_proba([sample])[0]
    confidence = max(probs)
    t1_edge = time.perf_counter()
    edge_latencies.append((t1_edge - t0_edge) * 1000)

    if confidence >= GAMMA:
        # High confidence: resolve locally at wire speed
        pred = int(np.argmax(probs))
        edge_preds.append(pred)
        edge_trues.append(true_label)
    else:
        # Ambiguous flow: escalate to Cloud under FHE
        t0_cloud = time.perf_counter()

        # Step A: Edge client encrypts continuous feature vector
        ctxt_x = HE.encrypt(sample.astype(np.float64))
        ctxt_w = HE.encrypt(cloud_weights)

        # Step B: Cloud evaluates linear dot product in ciphertext space
        ctxt_res = ctxt_x * ctxt_w

        # Step C: Edge receives ciphertext, decrypts locally with secret key
        decrypted_vector = HE.decrypt(ctxt_res)[:len(cloud_weights)]
        score = float(np.sum(decrypted_vector) + cloud_intercept)
        pred = 1 if score > 0 else 0

        t1_cloud = time.perf_counter()
        cloud_latencies.append((t1_cloud - t0_cloud) * 1000)

        # Track cloud predictions and verify plaintext vs. ciphertext score fidelity
        cloud_preds.append(pred)
        cloud_trues.append(true_label)

        plaintext_score = float(np.dot(sample, cloud_weights) + cloud_intercept)
        ckks_numerical_diffs.append(abs(score - plaintext_score))

    all_preds.append(pred)
    all_trues.append(true_label)

# ---------------------------------------------------------
# 4. Summary Performance Metrics
# ---------------------------------------------------------
total = len(X_test)
n_edge = len(edge_preds)
n_cloud = len(cloud_preds)

print("\n" + "=" * 62)
print("             SHIELD HYBRID PIPELINE RESULTS")
print("=" * 62)
print(f"Total Test Flows Evaluated      : {total}")
print(f"Resolved Locally at Edge        : {n_edge} ({n_edge / total * 100:.1f}%)")
print(f"Escalated to FHE Cloud Tier     : {n_cloud} ({n_cloud / total * 100:.1f}%)")
print("-" * 62)
print("EXECUTION LATENCIES:")
print(f"  Average Edge Plaintext Latency: {np.mean(edge_latencies):.4f} ms")
if cloud_latencies:
    print(f"  Average Cloud FHE Latency     : {np.mean(cloud_latencies):.4f} ms")
    speedup = np.mean(cloud_latencies) / np.mean(edge_latencies)
    print(f"  Edge Latency Advantage Ratio  : {speedup:.1f}x faster than FHE")
print("-" * 62)
print("CLASSIFICATION PERFORMANCE:")
print(f"  Overall Hybrid System Accuracy: {accuracy_score(all_trues, all_preds) * 100:.2f}%")
print(f"  Overall Hybrid System F1-Score: {f1_score(all_trues, all_preds):.4f}")

if n_edge > 0:
    print(f"  Edge Tier Accuracy (Easy)     : {accuracy_score(edge_trues, edge_preds) * 100:.2f}%")
if n_cloud > 0:
    print(f"  Cloud Tier Accuracy (Hard)    : {accuracy_score(cloud_trues, cloud_preds) * 100:.2f}%")
    print(f"  Max CKKS Floating-Point Noise : {max(ckks_numerical_diffs):.2e}")
print("=" * 62)