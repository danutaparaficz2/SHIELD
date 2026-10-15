import json
import os
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# Configuration & CPU Reference
# -------------------------------------------------------------
RESULTS_FILE = "latency_gpu_results.json"
OUTPUT_IMAGE = "fhe_benchmark_summary.png"
CPU_BASELINE_LATENCY_MS = 5.20  # Measured single-thread CPU baseline
CPU_THROUGHPUT = 1000.0 / CPU_BASELINE_LATENCY_MS

def load_data():
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"'{RESULTS_FILE}' not found. Run bench_latency_gpu.py first.")
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)
    return data

def main():
    data = load_data()

    # Extract metrics across batches
    batch_sizes = [v["batch_size"] for v in data.values()]
    kernel_ms = [v["median_kernel_ms"] for v in data.values()]
    h2d_ms = [v["median_h2d_ms"] for v in data.values()]
    d2h_ms = [v["median_d2h_ms"] for v in data.values()]
    total_ms = [v["total_latency_ms"] for v in data.values()]
    per_op_kernel_ms = [v["per_op_kernel_ms"] for v in data.values()]
    per_op_total_ms = [total_ms[i] / batch_sizes[i] for i in range(len(batch_sizes))]
    throughput = [v["throughput_ops_sec"] for v in data.values()]

    # Set overall aesthetic
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(wspace=0.3)

    # ---------------------------------------------------------
    # Panel 1: Per-Operation Latency
    # ---------------------------------------------------------
    axs[0].plot(batch_sizes, per_op_total_ms, marker='o', linewidth=2, color='#1f77b4', label='GPU Total (incl. PCIe)')
    axs[0].plot(batch_sizes, per_op_kernel_ms, marker='s', linestyle='--', linewidth=2, color='#2ca02c', label='GPU Pure Kernel')
    axs[0].axhline(y=CPU_BASELINE_LATENCY_MS, color='#d62728', linestyle=':', linewidth=2, label=f'CPU Baseline (~{CPU_BASELINE_LATENCY_MS} ms)')
    axs[0].set_title('Per-Operation Latency', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('Batch Size')
    axs[0].set_ylabel('Latency per op (ms)')
    axs[0].set_xscale('log', base=2)
    axs[0].set_xticks(batch_sizes)
    axs[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(fontsize=9)

    # ---------------------------------------------------------
    # Panel 2: Throughput Scaling
    # ---------------------------------------------------------
    axs[1].plot(batch_sizes, throughput, marker='o', linewidth=2, color='#2ca02c', label='GPU Throughput')
    axs[1].axhline(y=CPU_THROUGHPUT, color='#d62728', linestyle=':', linewidth=2, label=f'CPU Baseline (~{CPU_THROUGHPUT:.0f} ops/s)')
    axs[1].set_title('Throughput Scaling', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('Batch Size')
    axs[1].set_ylabel('Throughput (ops/sec)')
    axs[1].set_xscale('log', base=2)
    axs[1].set_xticks(batch_sizes)
    axs[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend(fontsize=9)

    s

    # ---------------------------------------------------------
    # Panel 3: Latency Breakdown (Stacked Bars)
    # ---------------------------------------------------------
    x = np.arange(len(batch_sizes))
    width = 0.45
    axs[2].bar(x, h2d_ms, width, label='Host-to-Device (PCIe)', color='#ff7f0e')
    axs[2].bar(x, kernel_ms, width, bottom=h2d_ms, label='GPU Kernel Compute', color='#1f77b4')
    axs[2].bar(x, d2h_ms, width, bottom=np.array(h2d_ms) + np.array(kernel_ms), label='Device-to-Host (PCIe)', color='#9467bd')
    axs[2].set_title('Batch Latency Breakdown', fontsize=12, fontweight='bold')
    axs[2].set_xlabel('Batch Size')
    axs[2].set_ylabel('Total Duration (ms)')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(batch_sizes)
    axs[2].grid(True, axis='y', linestyle='--', alpha=0.6)
    axs[2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"[✓] Benchmark visualization saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()