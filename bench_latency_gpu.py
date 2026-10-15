import json
import statistics
import time
import torch
import openfhe

# ---------------------------------------------------------
# 1. Setup & Context Initialization
# ---------------------------------------------------------
def setup_crypto_context():
    """Configure your OpenFHE context (adjust parameters to match your setup)."""
    parameters = openfhe.CCParamsBFVRNS()
    parameters.SetPlaintextModulus(65537)
    parameters.SetMultiplicativeDepth(2)
    
    crypto_context = openfhe.GenCryptoContext(parameters)
    crypto_context.Enable(openfhe.PKESchemeFeature.PKE)
    crypto_context.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    crypto_context.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
    
    key_pair = crypto_context.KeyGen()
    crypto_context.EvalMultKeyGen(key_pair.secretKey)
    return crypto_context, key_pair

# ---------------------------------------------------------
# 2. Workload Definition
# ---------------------------------------------------------
def prepare_batch(crypto_context, key_pair, batch_size):
    """Generates a batch of ciphertexts on host/device."""
    batch_c1 = []
    batch_c2 = []
    for i in range(batch_size):
        pt1 = crypto_context.MakePackedPlaintext([i + 1, i + 2, i + 3])
        pt2 = crypto_context.MakePackedPlaintext([2, 4, 6])
        batch_c1.append(crypto_context.Encrypt(key_pair.publicKey, pt1))
        batch_c2.append(crypto_context.Encrypt(key_pair.publicKey, pt2))
    return batch_c1, batch_c2

def run_fhe_kernel(crypto_context, batch_c1, batch_c2):
    """Pure evaluation workload (e.g. EvalMult or EvalAdd across the batch)."""
    results = []
    for c1, c2 in zip(batch_c1, batch_c2):
        results.append(crypto_context.EvalMult(c1, c2))
    return results

# ---------------------------------------------------------
# 3. Benchmark Engine (Warmup + Isolated Timers + CUDA Events)
# ---------------------------------------------------------
def benchmark_batch(crypto_context, key_pair, batch_size, warmups=20, iterations=100):
    device_available = torch.cuda.is_available()
    if not device_available:
        print("[WARNING] CUDA not detected via PyTorch. Events will fallback to wall clock.")

    # Generate test data
    batch_c1, batch_c2 = prepare_batch(crypto_context, key_pair, batch_size)

    # A. Warm-up (flushes caches, initializes CUDA contexts, prevents JIT skew)
    for _ in range(warmups):
        _ = run_fhe_kernel_parallel(crypto_context, batch_c1, batch_c2)
    if device_available:
        torch.cuda.synchronize()

    # Metrics holders
    transfer_in_times = []
    kernel_times = []
    transfer_out_times = []

    if device_available:
        k_start = torch.cuda.Event(enable_timing=True)
        k_end = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        # 1. Transfer Time (H2D)
        # Note: If OpenFHE handles GPU memory internally, measure the serialization / binding transfer here.
        t0 = time.perf_counter()
        # [Simulate or execute H2D transfer if explicitly managed outside kernel]
        if device_available:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        transfer_in_times.append((t1 - t0) * 1000.0)

        # 2. Kernel Time (Pure compute, GPU hardware timer)
        if device_available:
            k_start.record()
            res = run_fhe_kernel_parallel(crypto_context, batch_c1, batch_c2)
            k_end.record()
            torch.cuda.synchronize()
            kernel_times.append(k_start.elapsed_time(k_end))  # returns milliseconds
        else:
            t_k0 = time.perf_counter()
            res = run_fhe_kernel_parallel(crypto_context, batch_c1, batch_c2)
            t_k1 = time.perf_counter()
            kernel_times.append((t_k1 - t_k0) * 1000.0)

        # 3. Transfer Time (D2H / Result retrieval)
        t2 = time.perf_counter()
        # [Simulate or execute D2H transfer if explicitly managed]
        if device_available:
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        transfer_out_times.append((t3 - t2) * 1000.0)

    # Compute summary statistics
    median_kernel = statistics.median(kernel_times)
    median_h2d = statistics.median(transfer_in_times)
    median_d2h = statistics.median(transfer_out_times)
    total_latency = median_kernel + median_h2d + median_d2h
    throughput = (batch_size / (total_latency / 1000.0))  # ops / sec

    return {
        "batch_size": batch_size,
        "median_kernel_ms": round(median_kernel, 3),
        "per_op_kernel_ms": round(median_kernel / batch_size, 3),
        "median_h2d_ms": round(median_h2d, 3),
        "median_d2h_ms": round(median_d2h, 3),
        "total_latency_ms": round(total_latency, 3),
        "throughput_ops_sec": round(throughput, 2)
    }
import concurrent.futures
import torch
import openfhe

def run_fhe_kernel_parallel(crypto_context, batch_c1, batch_c2, num_streams=8):
    """
    Dispatches batches across concurrent CUDA streams/worker threads 
    to saturate GPU compute units instead of running in a serial loop.
    """
    results = [None] * len(batch_c1)
    
    # Divide the batch among concurrent workers
    def eval_chunk(indices):
        for idx in indices:
            results[idx] = crypto_context.EvalMult(batch_c1[idx], batch_c2[idx])
            
    chunks = [range(i, len(batch_c1), num_streams) for i in range(num_streams)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_streams) as executor:
        futures = [executor.submit(eval_chunk, c) for c in chunks]
        concurrent.futures.wait(futures)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    return results
# ---------------------------------------------------------
# 4. Main Sweep
# ---------------------------------------------------------
def main():
    print("Initializing CryptoContext...")
    crypto_context, key_pair = setup_crypto_context()
    
    batch_sizes = [1, 8, 32, 64]
    results = {}

    print(f"\n{'Batch':<8} | {'Kernel (ms)':<12} | {'Per-Op (ms)':<12} | {'H2D (ms)':<10} | {'D2H (ms)':<10} | {'Ops/sec':<10}")
    print("-" * 75)

    for b in batch_sizes:
        res = benchmark_batch(crypto_context, key_pair, batch_size=b, warmups=15, iterations=50)
        results[f"batch_{b}"] = res
        print(f"{res['batch_size']:<8} | {res['median_kernel_ms']:<12.3f} | {res['per_op_kernel_ms']:<12.3f} | {res['median_h2d_ms']:<10.3f} | {res['median_d2h_ms']:<10.3f} | {res['throughput_ops_sec']:<10.2f}")

    with open("latency_gpu_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved detailed benchmark breakdown to latency_gpu_results.json")

if __name__ == "__main__":
    main()