#!/bin/bash

repoDir=$(dirname "$(realpath "$0")")
echo "Repo: $repoDir"
cd $repoDir

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p data/gpu_output
mkdir -p data/reference_structures

# ── Step 1: Run CPU baseline (generates ground truth for 10 sequences) ────────
echo "--- Running CPU baseline ---"
chmod +x run_baseline.sh
./run_baseline.sh

# ── Step 2: Build GPU project ─────────────────────────────────────────────────
echo "--- Building GPU project ---"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

if [ $? -ne 0 ]; then
    echo "Build failed — stopping"
    exit 1
fi
cd $repoDir

# ── Step 3: Run GPU on same 10 sequences ──────────────────────────────────────
echo "--- Running GPU Nussinov on 10 sequences ---"
./build/nussinov \
    --sequence data/baseline_bins/seq_256.fa \
    --output   data/gpu_output/gpu_structures.txt \
    --maxSeqs  10 \
    --batchSize 10 \
    --numThreads 4

# ── Step 4: Validate scores ───────────────────────────────────────────────────
echo "--- Validating GPU scores against CPU reference ---"
python3 src/validate.py \
    --gpu data/gpu_output/gpu_structures.txt \
    --ref data/reference_structures/ref_seq_256.txt

# ── Step 5: Memory check (uncomment when needed) ──────────────────────────────
# echo "--- Running compute-sanitizer ---"
# compute-sanitizer ./build/nussinov \
#     --sequence  data/baseline_bins/seq_256.fa \
#     --output    data/gpu_output/gpu_structures_sanitizer.txt \
#     --maxSeqs   10 \
#     --batchSize 10 \
#     --numThreads 4

# ── Step 6: Full benchmark (uncomment after validation passes) ─────────────────
# echo "--- Full benchmark ---"
# nsys profile --stats=true ./build/nussinov \
#     --sequence  data/baseline_bins/seq_256.fa \
#     --output    data/gpu_output/gpu_structures_full.txt \
#     --maxSeqs   24685 \
#     --batchSize 1000 \
#     --numThreads 8