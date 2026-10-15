"""
SHIELD: Multi-Core FHE Scaling Benchmark on NVIDIA DGX Spark
Evaluates parallel batch processing throughput for ambiguous network flows.
"""

import time
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from Pyfhel import Pyfhel

plt.switch_backend('Agg')

print("=" * 65)
print(f"  SHIELD: DGX Spark High-Throughput FHE Scaling Benchmark")
print(f"  Available Hardware Cores: {os.cpu_count()}")
print("=" * 65)

# Setup CKKS Context
HE = Pyfhel()
HE.contextGen(scheme='CKKS', n=8192, scale=2**30, qi_sizes=[60, 30, 30, 60])
HE.keyGen()
HE.relinKeyGen()

# 10 continuous traffic features
N_FEATURES = 10
W = np.random.uniform(-1.0, 1.0, size=N_FEATURES).astype(np.float64)
b = 0.15

def run_fhe_inference_worker(batch_data):
    """Worker task simulating cloud evaluating a batch of encrypted flows."""
    results = []
    for sample in batch_data:
        ctxt_x = HE.encrypt(sample)
        ctxt_w = HE.encrypt(W)
        ctxt_res = ctxt_x * ctxt_w
        val = np.sum(HE.decrypt(ctxt_res)[:len(W)]) + b
        results.append(1 if val > 0 else 0)
    return len(results)

# 200 ambiguous flows offloaded for batched evaluation
N_SAMPLES = 200
synthetic_flows = [np.random.randn(N_FEATURES).astype(np.float64) for _ in range(N_SAMPLES)]

# Benchmark across core allocations up to host capacity
candidate_workers = [1, 2, 4, 8, 16]
worker_counts = [c for c in candidate_workers if c <= os.cpu_count()]

throughputs = []
total_times = []

print(f"\nProcessing {N_SAMPLES} encrypted flows across varying core allocations:")

for workers in worker_counts:
    chunk_size = int(np.ceil(N_SAMPLES / workers))
    batches = [synthetic_flows[i:i + chunk_size] for i in range(0, N_SAMPLES, chunk_size)]
    
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = list(executor.map(run_fhe_inference_worker, batches))
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    flows_per_sec = N_SAMPLES / elapsed
    
    total_times.append(elapsed)
    throughputs.append(flows_per_sec)
    print(f"  -> Cores: {workers:2d} | Elapsed: {elapsed:6.2f}s | Throughput: {flows_per_sec:6.1f} flows/sec")

# --- Generate Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

# Plot 1: Total Batch Processing Latency
ax1.plot(worker_counts, total_times, marker='o', color='#4C72B0', lw=2.5)
ax1.set_xlabel('Active Spark Compute Cores')
ax1.set_ylabel('Total Execution Time for 200 Flows (s)')
ax1.set_title('Batch Processing Latency')
ax1.grid(True, ls="--", alpha=0.6)

# Plot 2: Throughput Scaling
ax2.bar([str(w) for w in worker_counts], throughputs, color='#55A868', alpha=0.85, width=0.5)
ax2.set_xlabel('Active Spark Compute Cores')
ax2.set_ylabel('Throughput (Encrypted Flows / Second)')
ax2.set_title('DGX Spark Multi-Core Scaling')
ax2.grid(True, ls="--", alpha=0.6, axis='y')

plt.suptitle('SHIELD: DGX Spark Encrypted Cloud Scaling Performance', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('/workspace/shield_spark_scaling.png', bbox_inches='tight')
print("\nPlot saved successfully to /workspace/shield_spark_scaling.png")