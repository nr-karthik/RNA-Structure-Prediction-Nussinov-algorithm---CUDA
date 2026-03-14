# Scaling RNA Structure Prediction: Multi-Strategy CUDA Parallelization of the Nussinov Algorithm

**ECE 213 — Parallel Computing in Bioinformatics | UC San Diego | March 2026**  
**Team:** Karthik Nandagudi Raviprakash · Shan Ananth · Nisha Kumar

---

## Overview

This project parallelizes the Nussinov RNA secondary structure prediction algorithm using CUDA.
The algorithm solves an O(N³) dynamic programming problem — for a single sequence of 10,000
nucleotides, a CPU takes ~28 minutes. We implemented three CUDA kernel versions achieving up to
**761× speedup** over single-threaded CPU on an NVIDIA A30 GPU.

### Key Results (NVIDIA A30, kernel-only time via cudaEvents)

| Seq Length | Seqs   | CPU 1T   | V1 Kernel | V2 Kernel | V3 Kernel | V3 Speedup |
|:----------:|:------:|:--------:|:---------:|:---------:|:---------:|:----------:|
| ≤256 nt    | 24,685 | 9,120ms  | 261ms     | 2,590ms   | 2,540ms   | **3.6×**   |
| ≤512 nt    | 2,024  | 17,340ms | 406ms     | 1,826ms   | 1,752ms   | **9.9×**   |
| ≤4,096 nt  | 10     | 8,130ms  | 248ms     | 219ms     | 157ms     | **51.8×**  |
| ≤10,000 nt | 20     | ~34,240s | ~105s     | ~142s     | ~45s      | **761×**   |

### Kernel Versions

| Version | Strategy | Key Idea |
|---------|----------|----------|
| **V1** | Wavefront + Batch | 1 thread/cell, batch across sequences via `blockIdx.y`, mirror-write for coalesced reads |
| **V2** | Shared Memory + Tree Reduction | 1 block/cell, loads full DP rows into shared memory, log₂(512)=9 step tree reduction |
| **V3** | Diagonal-Adaptive Shared Memory | Loads only d+1 needed elements per diagonal, triples SM occupancy at large N |

---

## Repository Structure

```
.
├── src/
│   ├── main.cpp                  # Entry point, CLI argument parsing, batch processing loop
│   ├── nussinov.cu               # All CUDA kernels (V1, V2, V3, Sequential) + host logic
│   ├── nussinov.cuh              # GpuNussinov class declaration, kernel prototypes
│   ├── nussinov_cpu.cpp          # Single-threaded CPU reference implementation
│   ├── benchmark_cpu_omp.cpp     # OpenMP multi-threaded CPU benchmark
│   ├── gpuProperty.cu            # Prints GPU device properties at startup
│   ├── validate.py               # Compares GPU scores against CPU reference
│   ├── timer.hpp                 # Wall-clock timing utilities
│   └── kseq.h                    # Lightweight FASTA/FASTQ parser
├── data/
│   ├── baseline_bins/            # Input sequence bins — included in repo
│   │   ├── seq_256.fa            # 24,685 sequences ≤256 nt
│   │   ├── seq_512.fa            # 2,024 sequences ≤512 nt
│   │   ├── seq_1024.fa           # sequences ≤1,024 nt
│   │   ├── seq_4096.fa           # 10 sequences ≤4,096 nt
│   │   ├── seq_8192.fa           # sequences ≤8,192 nt
│   │   └── seq_10000.fa          # 20 sequences ≤10,000 nt
│   └── reference_structures/     # Pre-generated CPU reference outputs — included in repo
│       ├── ref_seq_256.txt
│       ├── ref_seq_512.txt
│       ├── ref_seq_1024.txt
│       └── ref_seq_10000.txt
├── reports/                      # Nsight Systems profiling output (text reports)
├── bin_sequences.py              # Script to bin bpRNA sequences by length
├── CMakeLists.txt                # CMake build configuration
├── run.sh                        # Main script: build + validate + full benchmark
├── run_baseline.sh               # Regenerates CPU reference structures
└── README.md                     # This file
```

---

## Environment Setup & Reproducing Results

### Prerequisites

You need access to UCSD's DSMLP cluster with an NVIDIA A30 GPU node.

---

### Step 1 — Clone the Repository

SSH into the DSMLP login node, then clone the repo into your home directory:

```bash
ssh <your-username>@dsmlp-login.ucsd.edu

git clone https://github.com/<your-username>/RNA-Structure-Prediction-Nussinov-algorithm---CUDA.git
cd RNA-Structure-Prediction-Nussinov-algorithm---CUDA
```

The repository already includes:
- `data/baseline_bins/` — all input sequence bins
- `data/reference_structures/` — pre-generated CPU reference outputs for validation

No dataset download or separate CPU baseline step is required.

---

### Step 2 — Launch a GPU Node

From the DSMLP login node:

```bash
/opt/launch-sh/bin/launch.sh \
    -v a30 \
    -c 8 \
    -g 1 \
    -m 8 \
    -i yatisht/ece213-wi26:latest
```

You are now inside a container on an A30 node. Navigate to your repo:

```bash
cd ~/RNA-Structure-Prediction-Nussinov-algorithm---CUDA
```

---

### Step 3 — Run the Full Pipeline

```bash
chmod +x run.sh run_baseline.sh
./run.sh
```

This script does the following in order:

1. **Runs the CPU baseline** — regenerates reference structures (already included in repo, this step verifies them)
2. **Builds the GPU project** — compiles V1, V2, V3, and Sequential kernels via CMake
3. **Validates V1, V2, V3** — compares GPU scores against CPU reference on 10 sequences
4. **Runs the full benchmark** — all three versions across all four sequence bins

Expected validation output:

```
--- Validating V1 scores ---
[PASS] bpRNA_CRW_1897    score=52
[PASS] bpRNA_CRW_4052    score=103
...
Result: 10/10 sequences passed
✅ All scores match — GPU kernel is correct

--- Validating V2 scores ---
...
✅ All scores match — GPU kernel is correct

--- Validating V3 scores ---
...
✅ All scores match — GPU kernel is correct
```

Expected benchmark output (kernel-only times, A30):

```
[V1] seq_256   → ~261ms       [V2] seq_256   → ~2,590ms    [V3] seq_256   → ~2,540ms
[V1] seq_512   → ~406ms       [V2] seq_512   → ~1,826ms    [V3] seq_512   → ~1,752ms
[V1] seq_4096  → ~248ms       [V2] seq_4096  → ~219ms      [V3] seq_4096  → ~157ms
[V1] seq_10000 → ~105,000ms   [V2] seq_10000 → ~142,000ms  [V3] seq_10000 → ~45,000ms
```

---

### Alternative: Launch Node and Auto-Run Everything

From the DSMLP login node, this does Steps 2 and 3 in a single command:

```bash
/opt/launch-sh/bin/launch.sh \
    -v a30 \
    -c 8 \
    -g 1 \
    -m 8 \
    -i yatisht/ece213-wi26:latest \
    -f ~/RNA-Structure-Prediction-Nussinov-algorithm---CUDA/run.sh
```

---

## Manual Usage

### Build Only

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..
```

Built targets:
- `build/nussinov` — main GPU executable
- `build/nussinov_baseline` — CPU reference implementation
- `build/benchmark_cpu_omp` — OpenMP multi-threaded CPU benchmark

### Run a Single GPU Kernel

```bash
./build/nussinov \
    --sequence   data/baseline_bins/seq_4096.fa \
    --output     data/gpu_output/output.txt \
    --maxSeqs    10 \
    --batchSize  10 \
    --numThreads 4 \
    --version    4
#   version: 1=V1  2=V2  3=Sequential  4=V3
```

**All flags:**

| Flag            | Description                                      | Default  |
|-----------------|--------------------------------------------------|----------|
| `--sequence`    | Path to input `.fa` file                         | required |
| `--output`      | Path to write predicted structures               | required |
| `--maxSeqs`     | Max sequences to process                         | required |
| `--batchSize`   | Sequences per GPU batch (affects memory usage)   | 1000     |
| `--numThreads`  | CPU threads for I/O and traceback                | 4        |
| `--version`     | Kernel: `1`=V1, `2`=V2, `3`=Sequential, `4`=V3  | 1        |

### Validate GPU Output Against CPU Reference

```bash
python3 src/validate.py \
    --gpu data/gpu_output/output.txt \
    --ref data/reference_structures/ref_seq_256.txt
```

### Regenerate CPU Reference Structures

Only needed if you want to regenerate from scratch:

```bash
./run_baseline.sh
# Writes to data/reference_structures/ref_seq_256.txt etc.
```

### OpenMP CPU Benchmark

```bash
./build/benchmark_cpu_omp \
    data/baseline_bins/seq_4096.fa \
    10 \
    8
# Args: input_file  maxSeqs  numThreads
```

---

## Dataset

Sequences are from [bpRNA](https://bprna.cgrb.oregonstate.edu/), a curated dataset of 28,370 RNA
sequences with experimentally verified secondary structures. The pre-binned `.fa` files are
included directly in `data/baseline_bins/`.

If you want to regenerate the bins from the full bpRNA dataset:

```bash
# Place bpRNA_1m_90.fasta in data/
python3 bin_sequences.py
# Generates seq_256.fa, seq_512.fa, seq_1024.fa, seq_4096.fa, seq_8192.fa, seq_10000.fa
# in data/baseline_bins/
```

### Input Format (standard FASTA, RNA bases only)

```
>bpRNA_CRW_1897
GGCUUCAUAGUUCAGUCGGUAGAACGGCGGACUGAAAUCUCGUAGUCGUGGGUUCGAGUCCCACUUGAGCC
```

---

## Implementation Details

### Algorithm

The Nussinov algorithm fills an N×N DP table in diagonal wavefront order:

```
dp[i][j] = max(
    dp[i+1][j-1] + 1                      if seq[i] pairs with seq[j]   ← base pair
    max_k( dp[i][k] + dp[k+1][j] )  for k in [i, j)                     ← bifurcation
)
```

Valid base pairs: A-U, U-A, G-C, C-G, G-U, U-G (Watson-Crick + wobble)

### Key Optimizations

**Coalesced memory access (V1, V2, V3)**  
Mirror-write `dp[j][i] = dp[i][j]` converts strided column reads into consecutive row reads,
reducing warp memory transactions from 32 to 1.

**Batch parallelism (V1, V2, V3)**  
Grid dimension Y (`blockIdx.y`) indexes sequences, processing thousands simultaneously in one
kernel launch. This keeps the GPU saturated even on short diagonals with few cells.

**Shared memory tiling (V2)**  
Each block loads the two required DP rows (`row_i`, `row_j`) into shared memory (40KB each),
so all bifurcation reads hit shared memory at 30-cycle latency instead of global memory at
400-cycle latency.

**Diagonal-adaptive loading (V3)**  
V2 always loads full N-length rows. V3 loads only the d+1 elements actually needed for diagonal
d. At d = N/2 = 5,000 for N=10,000, each block uses (2×5,001 + 512)×4 = 42KB instead of 82KB,
allowing 3 blocks per SM instead of 1 — tripling SM occupancy from 25% to 75%.

**cudaFuncSetAttribute for large shared memory**  
CUDA defaults to a 48KB shared memory limit per block. V2 and V3 require up to 82KB at
N=10,000. We call `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)` before
each kernel to opt into the A30's full 99KB hardware limit. Without this, V2/V3 crash with
`cudaErrorInvalidValue` for sequences longer than 6,000 nt.

---

## Profiling Notes

Nsight Systems profiles are in `reports/`. Key findings from `nsys_v2_4096.txt`:

```
Total program:         733ms
Kernel execution:      420ms  (57%) — matches cudaEvent timing of 422ms ✓
cudaLaunchKernel:      234ms  (32%) — 1,548 launches × 151μs each
cudaDeviceSynchronize: 187ms  (25%)
cudaMalloc:            140ms  (19%)
```

> Note: These profiles were captured on a MIG 2g.12gb instance (1/3 of A30), which adds
> virtualization overhead (~151μs per kernel launch vs ~5μs on the full A30). Full A30
> benchmark results are in `data/gpu_output/`.

---

## References

1. R. Nussinov et al., "Algorithms for Loop Matchings," SIAM J. Applied Mathematics, 1978.
2. S. Danaee et al., "bpRNA: large-scale automated annotation of RNA secondary structure," NAR, 2018.
3. C. Tan et al., "GPU Implementation of Nussinov Dynamic Programming," IEEE ISSPIT, 2010.
4. NVIDIA, "CUDA C++ Programming Guide," v12.0, 2023.
5. NVIDIA, "Nsight Systems User Guide," 2023.