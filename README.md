# SHIELD: Privacy-Preserving Network-Flow Analysis

## 1. SHIELD at a glance

SHIELD is a research prototype for hybrid edge/cloud intrusion detection. It
aims to resolve high-confidence network-flow predictions locally and send
ambiguous cases to a remote evaluator as encrypted data. The repository
contains synthetic and NSL-KDD classification experiments, CKKS/OpenFHE
examples, and exploratory CPU/GPU benchmarks for NVIDIA DGX Spark.

**Current status:** feasibility work, not a production system. The project
demonstrates encrypted arithmetic and hybrid-routing prototypes, but does not
yet establish a production edge/cloud service or Blackwell-accelerated FHE.

## 2. Architecture and research questions

Intended workflow:

1. Extract numeric features from network flows.
2. Score flows locally; resolve high-confidence predictions at the edge.
3. Encrypt ambiguous flows with CKKS and evaluate them without exposing the
   plaintext features to the compute tier.
4. Return encrypted results for decryption and classification by the edge.

DGX Spark provides a useful development platform for this design: its Grace
Arm CPU can run preprocessing and CPU-capable FHE libraries; its Blackwell GPU
can potentially accelerate suitably parallelized polynomial operations; and
NVLink-C2C unified memory can simplify CPU/GPU data sharing in an implementation
that supports it. These benefits depend on the software actually using the
GPU and memory path—CUDA availability alone does not accelerate OpenFHE.

The main research questions are:

- How much traffic can a lightweight local model resolve while maintaining
  useful detection quality?
- What are the latency and throughput costs of evaluating ambiguous
  predictions with CKKS?
- Can a verified GPU FHE backend improve throughput over a comparable CPU
  implementation after accounting for transfers and batching?

## 3. Experiments and datasets

### Synthetic hybrid prototype

[`shield_hybrid_demo.py`](./shield_hybrid_demo.py) creates 1,000 synthetic
examples with 10 continuous features, trains logistic regression, and routes
flows at a confidence threshold of 0.75. Ambiguous cases use a Pyfhel CKKS
encrypted dot product. It reports routing counts, latency, accuracy, F1, and
approximation error. This tests the control flow, not detection on real traffic.

### NSL-KDD benchmark

[`shield_real_data_benchmark.py`](./shield_real_data_benchmark.py) downloads
NSL-KDD `KDDTrain+.csv` on first use and caches it at
`/workspace/nsl_kdd_train.csv`. It binarizes normal/attack labels, selects 10
numeric features, and samples 2,500 training and 500 evaluation rows. A linear
logistic-regression model handles high-confidence flows at the edge; ambiguous
flows are transformed into 65 degree-2 polynomial features and evaluated with
a second model's weights using Pyfhel CKKS.

This is a dataset-based feasibility experiment: the scaler is currently fit
before the train/evaluation split, the sample is drawn from the training file
without stratification, and model training/feature transforms run in plaintext.
Encryption and decryption happen in the same process; it does not measure a
separate edge/cloud deployment or end-to-end latency. It prints metrics but
does not save them or generate a chart.

### Cryptography and hardware experiments

- [`test_openfhe_client.py`](./test_openfhe_client.py) is an OpenFHE CKKS
  encrypt/multiply/decrypt smoke test.
- [`spark_scale_benchmark.py`](./spark_scale_benchmark.py) explores a Pyfhel
  CKKS workload over multiple CPU process-worker counts.
- [`bench_latency_gpu.py`](./bench_latency_gpu.py) times OpenFHE `EvalMult`
  calls configured with BFV. The code does not dispatch these calls to CUDA,
  and its transfer timing sections are placeholders.
- [`fhe_blackwell_unified_benchmark.py`](./fhe_blackwell_unified_benchmark.py)
  compares CPU/GPU floating-point FFT convolution. It is a CUDA compute proxy,
  not CKKS arithmetic; `.to(device)` copies host tensors to the GPU, so it does
  not demonstrate zero-copy operation.

These scripts use different schemes, parameters, and workloads; their numbers
are not directly comparable.

## 4. Results

Only two figures are shown here. The latency charts and the Blackwell FFT proxy
are exploratory and live in [visualizations/README.md](./visualizations/README.md);
the 195× edge/cloud ratio is not reproducible and should not be cited.

**Hybrid architecture** (design, not a measurement):

![SHIELD hybrid architecture](./visualizations/shield_architecture.svg)

**DGX Spark CPU scaling** (exploratory, system-specific):

![Spark CPU scaling](./visualizations/shield_spark_scaling.png)

Pyfhel CKKS (n=8192) over 200 flows with 1–16 process workers. This is a CPU
benchmark; raw result samples are not checked in, so treat the numbers as
indicative only.

## 5. DGX Spark and Docker setup

### What the image builds

[`Dockerfile`](./Dockerfile) is a development-container recipe, not a
ready-to-run SHIELD image. It uses NVIDIA CUDA 12.8 on Ubuntu 22.04, installs
build tools, Python 3, pybind11 and OpenMP development files, then builds
OpenFHE 1.5.1 (Release, static libraries, OpenMP) and its Python bindings
(revision not pinned). It does not install the project's Python packages
(PyTorch, Pyfhel, NumPy, pandas, scikit-learn, Matplotlib) or copy the scripts;
it starts Bash in `/workspace`.

### What has been verified

| Status | Item | Basis |
| --- | --- | --- |
| Verified on Spark (from saved outputs) | CUDA PyTorch ran on an NVIDIA GB10 device | [Blackwell FFT proxy chart](./visualizations/shield_blackwell_acceleration.png) title |
| Verified on Spark (from saved outputs) | OpenFHE Python bindings ran CPU-side batched `EvalMult` timings | JSON snapshots in [`visualizations/`](./visualizations) |
| Verified on Spark (from saved outputs) | Pyfhel CKKS multi-process scaling | [Spark scaling chart](./visualizations/shield_spark_scaling.png) |
| Not recorded | Clean `docker build` log, host driver/toolkit versions, exact package versions | No log or lockfile in the repository |
| Not established | GPU backend built, linked, or used by OpenFHE | The Dockerfile only attempts a clone that is allowed to fail and never builds it; no profiling shows OpenFHE CUDA kernels |

The "verified" rows are inferred from checked-in outputs, not from a recorded
build/test session. Add the build log and the versions used when re-running.

### Build and run

Mounting a host folder over `/workspace` hides the backend clone stored there,
so the example mounts the project elsewhere. Build once and reuse the image:

```bash
docker build -t shield-spark .
docker run --rm -it --gpus all \
  -v "$PWD:/work/shield" -w /work/shield \
  shield-spark
```

GPU access requires a working host driver and NVIDIA Container Toolkit. Inside
the container, check device visibility (`nvidia-smi`, and
`torch.cuda.is_available()` if PyTorch is installed) and each script's imports
before running. Install and pin dependencies for repeatable runs.

For meaningful hardware comparisons, use the same FHE scheme, parameters,
ciphertext sizes, and operations on CPU and GPU; confirm GPU kernels are
actually launched; measure encryption, evaluation, decryption, and transfers
separately; and record device, software, and raw samples. Set and record
OpenMP/process worker counts to avoid oversubscription.

## 6. Current limitations and next steps

**Demonstrated:** synthetic and NSL-KDD-based classification experiments,
CKKS encrypt/evaluate/decrypt examples, CPU process scaling, and exploratory
CUDA/OpenFHE timing scripts.

**Not yet established:** generalization to independent or live network
traffic, unbiased NSL-KDD evaluation, a separate edge/cloud trust boundary,
production key management/security, or real FHE acceleration on Blackwell.
The container's CUDA base and attempted backend clone do not prove GPU FHE
execution.

**Next milestone:** correct the NSL-KDD preprocessing/split, then benchmark the
same real FHE operation on CPU and a verified GPU backend, validate outputs
against a reference, and publish raw measurements with reproducible setup.

## 7. Repository contents and commands

| File | Purpose |
| --- | --- |
| [`shield_hybrid_demo.py`](./shield_hybrid_demo.py) | Synthetic edge/cloud CKKS prototype. |
| [`shield_real_data_benchmark.py`](./shield_real_data_benchmark.py) | NSL-KDD asymmetric edge/cloud benchmark. |
| [`test_openfhe_client.py`](./test_openfhe_client.py) | OpenFHE CKKS smoke test. |
| [`spark_scale_benchmark.py`](./spark_scale_benchmark.py) | CPU process-scaling experiment; produces the Spark scaling chart. |
| [`fhe_blackwell_unified_benchmark.py`](./fhe_blackwell_unified_benchmark.py) | CPU/GPU floating-point FFT proxy benchmark. |
| [`bench_latency_gpu.py`](./bench_latency_gpu.py) | OpenFHE timing experiment and JSON output. |
| [`plot_latency.py`](./plot_latency.py) | Generates the batch timing summary chart. |
| [`Dockerfile`](./Dockerfile) | CUDA/OpenFHE development container recipe. |
| [`visualizations/`](./visualizations) | Saved charts and JSON snapshots. |

The repository has no Python dependency lockfile or unified setup command.
Subject to installing the required packages and native libraries, run
individual experiments with:

```bash
python3 shield_hybrid_demo.py
python3 shield_real_data_benchmark.py
python3 test_openfhe_client.py
python3 spark_scale_benchmark.py
python3 fhe_blackwell_unified_benchmark.py
python3 bench_latency_gpu.py
```

The NSL-KDD script downloads its dataset on first run. Benchmark outputs may
overwrite saved snapshots. `plot_latency.py` expects a compatible
`latency_gpu_results.json` in its working directory; the checked-in plot and
JSON snapshots are not all verified to come from the same run.
