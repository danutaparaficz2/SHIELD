# Experimental visualizations

These charts are exploratory and are not part of the main README's claims.
None demonstrates a validated GPU-accelerated FHE result. Raw samples and run
provenance are not checked in for most of them.

| Image | What it shows | Evidence status |
| --- | --- | --- |
| ![Edge/cloud latency](./latency_comparison.svg) | Local plaintext vs. cloud CKKS latency; cloud samples are described as 1,000 iterations. | Edge values and the displayed ratio are visual estimates; not reproducible. Do not cite the ratio. |
| ![GPU latency distribution](./latency_gpu_plot.svg) | Median 5.260 ms, IQR 5.050–5.485 ms. | Snapshot only; no raw samples. |
| ![CPU/GPU latency comparison](./cpu_gpu_latency_comparison.svg) | GPU median vs. a CPU value estimated at about 5.20 ms. | Not a validated CPU/GPU FHE comparison. |
| ![Batched operation summary](./fhe_benchmark_summary.png) | Per-operation latency, throughput and transfer/kernel timing, batches 1–64. | The two JSON snapshots disagree; transfer timing is a placeholder; GPU FHE execution is not established. |
| ![Blackwell FFT proxy](./shield_blackwell_acceleration.png) | CPU/GPU float32 FFT convolution timings and speedups. | Floating-point CUDA proxy, not CKKS. |
