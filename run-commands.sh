#!/bin/bash

repoDir=$(dirname "$(realpath "$0")")
echo "Repo: $repoDir"
cd $repoDir

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p data/gpu_output
mkdir -p data/reference_structures

# ── Step 1: CPU baseline ──────────────────────────────────────────────────────
echo "--- Running CPU baseline ---"
chmod +x run_baseline.sh
./run_baseline.sh

# ── Step 2: Build ─────────────────────────────────────────────────────────────
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

# ── Step 3: Validate V1, V2, V3 on 10 sequences ──────────────────────────────
echo "--- Validating V1 ---"
./build/nussinov \
    --sequence   data/baseline_bins/seq_256.fa \
    --output     data/gpu_output/gpu_v1_val.txt \
    --maxSeqs    10 \
    --batchSize  10 \
    --numThreads 4 \
    --version    1

echo "--- Validating V2 ---"
./build/nussinov \
    --sequence   data/baseline_bins/seq_256.fa \
    --output     data/gpu_output/gpu_v2_val.txt \
    --maxSeqs    10 \
    --batchSize  10 \
    --numThreads 4 \
    --version    2

echo "--- Validating V3 ---"
./build/nussinov \
    --sequence   data/baseline_bins/seq_256.fa \
    --output     data/gpu_output/gpu_v3_val.txt \
    --maxSeqs    10 \
    --batchSize  10 \
    --numThreads 4 \
    --version    4

# ── Step 4: Validate scores against CPU reference ─────────────────────────────
echo "--- Validating V1 scores ---"
python3 src/validate.py \
    --gpu data/gpu_output/gpu_v1_val.txt \
    --ref data/reference_structures/ref_seq_256.txt

echo "--- Validating V2 scores ---"
python3 src/validate.py \
    --gpu data/gpu_output/gpu_v2_val.txt \
    --ref data/reference_structures/ref_seq_256.txt

echo "--- Validating V3 scores ---"
python3 src/validate.py \
    --gpu data/gpu_output/gpu_v3_val.txt \
    --ref data/reference_structures/ref_seq_256.txt

# ── Step 5: Full benchmark — V1, V2, V3 across all sequence bins ──────────────
echo "========================================="
echo "--- Full Benchmark ---"
echo "========================================="

VERSIONS=("1" "2" "3")
VERSION_NAMES=("V1" "V2" "V3")
SEQ_BINS=("seq_256.fa 24685" "seq_512.fa 2024" "seq_4096.fa 10" "seq_10000.fa 20")

for i in "${!VERSIONS[@]}"; do
    ver=${VERSIONS[$i]}
    name=${VERSION_NAMES[$i]}
    echo ""
    echo "--- Benchmarking $name ---"
    for entry in "${SEQ_BINS[@]}"; do
        file=$(echo $entry | cut -d' ' -f1)
        maxseqs=$(echo $entry | cut -d' ' -f2)
        bin="${file%.fa}"
        echo "  [$name] $bin ($maxseqs seqs)"
        ./build/nussinov \
            --sequence   data/baseline_bins/$file \
            --output     data/gpu_output/bench_${name}_${bin}.txt \
            --maxSeqs    $maxseqs \
            --batchSize  1000 \
            --numThreads 8 \
            --version    $ver
    done
done

# ── Step 6: Memory check — uncomment when needed ──────────────────────────────
# echo "--- compute-sanitizer on V3 ---"
# compute-sanitizer ./build/nussinov \
#     --sequence   data/baseline_bins/seq_256.fa \
#     --output     data/gpu_output/gpu_v3_sanitizer.txt \
#     --maxSeqs    10 \
#     --batchSize  10 \
#     --numThreads 4 \
#     --version    4

# ── Step 7: Nsight profiling — uncomment when needed ──────────────────────────
# echo "--- nsys profile V2 seq_4096 ---"
# nsys profile --stats=true ./build/nussinov \
#     --sequence   data/baseline_bins/seq_4096.fa \
#     --output     data/gpu_output/gpu_v2_profile.txt \
#     --maxSeqs    10 \
#     --batchSize  10 \
#     --numThreads 8 \
#     --version    2

# echo "--- nsys profile V3 seq_4096 ---"
# nsys profile --stats=true ./build/nussinov \
#     --sequence   data/baseline_bins/seq_4096.fa \
#     --output     data/gpu_output/gpu_v3_profile.txt \
#     --maxSeqs    10 \
#     --batchSize  10 \
#     --numThreads 8 \
#     --version    4