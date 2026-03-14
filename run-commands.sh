# #!/bin/sh

# # Change directory (DO NOT CHANGE!)
# repoDir=$(dirname "$(realpath "$0")")
# echo $repoDir
# cd $repoDir

# mkdir -p build
# cd build
# cmake ..
# make -j4
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/tbb_cmake_build/tbb_cmake_build_subdir_release


# # nsys profile --stats=true 
# ## Basic run to align the first 10 pairs (20 sequences) in the sequences.fa in batches of 2 reads
# ## HINT: may need to change values for the assignment tasks. You can create a sequence of commands
# # ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --batchSize 1000 --output alignment.fa --numThreads 8

# ## Debugging with Compute Sanitizer
# ## Run this command to detect illegal memory accesses (out-of-bounds reads/writes) and race conditions.
# compute-sanitizer ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --batchSize 1000 --output alignment.fa --numThreads 8

# ## For debugging and evaluate the alignment accuracy
# ./check_alignment --raw ../data/sequences.fa --alignment alignment.fa       # add -v to see failed result instead of a summary
# ./compare_alignment --reference ../data/reference_alignment.fa --estimate alignment.fa # add -v to see pair-wise comparisons

#!/bin/sh

# repoDir=$(dirname "$(realpath "$0")")
# echo $repoDir
# cd $repoDir

# mkdir -p build
# cd build
# cmake ..
# make -j4
# cd ..

# mkdir -p data/gpu_output

# echo "=== V2 seq_10000 20 sequences ==="
# ./build/nussinov --sequence data/baseline_bins/seq_10000.fa \
#     --output data/gpu_output/gpu_v2_10000.txt \
#     --maxSeqs 20 --batchSize 20 --numThreads 8 --version 2



#!/bin/sh

repoDir=$(dirname "$(realpath "$0")")
echo $repoDir
cd $repoDir

rm -rf build
mkdir -p build
cd build
cmake ..
make -j4
cd ..

mkdir -p data/gpu_output
mkdir -p reports

echo "=== NCU Roofline V2 seq_256 ==="
ncu --set roofline \
    --kernel-name nussinov_kernel_v2 \
    ./build/nussinov \
    --sequence data/baseline_bins/seq_256.fa \
    --output /dev/null \
    --maxSeqs 10 --batchSize 10 --numThreads 8 --version 2 \
    2>&1 | tee reports/ncu_roofline_v2_256.txt

echo "=== NCU Full V2 seq_256 ==="
ncu --set full \
    --kernel-name nussinov_kernel_v2 \
    ./build/nussinov \
    --sequence data/baseline_bins/seq_256.fa \
    --output /dev/null \
    --maxSeqs 10 --batchSize 10 --numThreads 8 --version 2 \
    2>&1 | tee reports/ncu_full_v2_256.txt

echo "=== NSYS V2 seq_4096 ==="
nsys profile \
    --output reports/nsys_v2_4096 \
    --stats=true \
    ./build/nussinov \
    --sequence data/baseline_bins/seq_4096.fa \
    --output /dev/null \
    --maxSeqs 10 --batchSize 10 --numThreads 8 --version 2 \
    2>&1 | tee reports/nsys_v2_4096.txt