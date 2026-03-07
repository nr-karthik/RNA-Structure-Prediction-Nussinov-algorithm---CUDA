#include "nussinov.cuh"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <cuda_runtime.h>

// ─── Device function ────────────────────────────────────────────────────────

__device__ bool complementary(char a, char b) {
    return ((a == 'G' && b == 'U') || (a == 'U' && b == 'G') ||
            (a == 'A' && b == 'U') || (a == 'U' && b == 'A') ||
            (a == 'C' && b == 'G') || (a == 'G' && b == 'C'));
}

// ─── Host complementary (for traceback on CPU) ──────────────────────────────

static bool complementary_host(char a, char b) {
    return ((a == 'G' && b == 'U') || (a == 'U' && b == 'G') ||
            (a == 'A' && b == 'U') || (a == 'U' && b == 'A') ||
            (a == 'C' && b == 'G') || (a == 'G' && b == 'C'));
}

// ─── Kernel ─────────────────────────────────────────────────────────────────

__global__ void nussinov_kernel(
    int32_t* d_dp,
    int32_t* d_seqLen,
    char*    d_seqs,
    int32_t  longestLen,
    int      diagonal)
{
    int seq_idx   = blockIdx.y;
    int i         = blockIdx.x * blockDim.x + threadIdx.x;
    int sizeOfSeq = d_seqLen[seq_idx];

    if (diagonal >= sizeOfSeq) return;
    if (i >= sizeOfSeq - diagonal) return;

    int j = i + diagonal;

    char*    seq = d_seqs + seq_idx * longestLen;
    int32_t* dp  = d_dp  + seq_idx * longestLen * longestLen;

    int result = 0;

    // Case A: i and j form a base pair
    if (complementary(seq[i], seq[j]))
        result = max(result, dp[(i+1)*longestLen + (j-1)] + 1);

    // Case B: bifurcation (absorbs unpaired cases at boundaries)
    for (int k = i; k < j; k++)
        result = max(result, dp[i*longestLen + k] + dp[(k+1)*longestLen + j]);

    dp[i*longestLen + j] = result;
    dp[j*longestLen + i] = result;
}

// ─── allocateMem ────────────────────────────────────────────────────────────

void GpuNussinov::allocateMem() {
    auto customComparator = [](const RNASequence& a, const RNASequence& b) {
        return a.seq.size() < b.seq.size();
    };
    longestLen = std::max_element(seqs.begin(), seqs.end(), customComparator)->seq.size();

    cudaError_t err;

    err = cudaMalloc(&d_seqs, numSeqs * longestLen * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU Error allocating d_seqs: %s (%s)\n",
                cudaGetErrorString(err), cudaGetErrorName(err)); exit(1);
    }

    err = cudaMalloc(&d_seqLen, numSeqs * sizeof(int32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU Error allocating d_seqLen: %s (%s)\n",
                cudaGetErrorString(err), cudaGetErrorName(err)); exit(1);
    }

    err = cudaMalloc(&d_dp, numSeqs * longestLen * longestLen * sizeof(int32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU Error allocating d_dp: %s (%s)\n",
                cudaGetErrorString(err), cudaGetErrorName(err)); exit(1);
    }
    cudaMemset(d_dp, 0, numSeqs * longestLen * longestLen * sizeof(int32_t));

    err = cudaMalloc(&d_structures, numSeqs * longestLen * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU Error allocating d_structures: %s (%s)\n",
                cudaGetErrorString(err), cudaGetErrorName(err)); exit(1);
    }
    cudaMemset(d_structures, '.', numSeqs * longestLen * sizeof(char));
}

// ─── transferSequencesToDevice ───────────────────────────────────────────────

void GpuNussinov::transferSequencesToDevice() {
    cudaError_t err;

    std::vector<char> host_sequences(longestLen * numSeqs, 0);
    for (int i = 0; i < numSeqs; i++)
        std::memcpy(host_sequences.data() + i * longestLen,
                    seqs[i].seq.data(), seqs[i].seq.size());

    err = cudaMemcpy(d_seqs, host_sequences.data(),
                     numSeqs * longestLen * sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU Error transferring sequences: %s (%s)\n",
                cudaGetErrorString(err), cudaGetErrorName(err)); exit(1);
    }

    std::vector<int32_t> host_seqLen(numSeqs);
    for (int i = 0; i < numSeqs; i++)
        host_seqLen[i] = seqs[i].seq.size();

    err = cudaMemcpy(d_seqLen, host_seqLen.data(),
                     numSeqs * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU Error transferring lengths: %s (%s)\n",
                cudaGetErrorString(err), cudaGetErrorName(err)); exit(1);
    }
}

// ─── transferDPMatrixToHost ──────────────────────────────────────────────────

DP_PATH GpuNussinov::transferDPMatrixToHost() {
    DP_PATH dp_matrix(longestLen * longestLen * numSeqs, 0);

    cudaError_t err = cudaMemcpy(dp_matrix.data(), d_dp,
                                  longestLen * longestLen * numSeqs * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU Error transferring DP matrix to host: %s (%s)\n",
                cudaGetErrorString(err), cudaGetErrorName(err)); exit(1);
    }
    return dp_matrix;
}

// ─── traceback (recursive helper) ───────────────────────────────────────────

static void traceback_recursive(
    int i, int j,
    const std::string& seq,
    const DP_PATH& dp,
    int32_t L,           // longestLen — stride for indexing
    std::string& structure)
{
    if (i >= j) return;

    // Case 1: i is unpaired
    if (dp[i*L + j] == dp[(i+1)*L + j]) {
        traceback_recursive(i+1, j, seq, dp, L, structure);
    }
    // Case 2: j is unpaired
    else if (dp[i*L + j] == dp[i*L + (j-1)]) {
        traceback_recursive(i, j-1, seq, dp, L, structure);
    }
    // Case 3: i and j are paired with each other
    else if (complementary_host(seq[i], seq[j]) &&
             dp[i*L + j] == dp[(i+1)*L + (j-1)] + 1) {
        structure[i] = '(';
        structure[j] = ')';
        traceback_recursive(i+1, j-1, seq, dp, L, structure);
    }
    // Case 4: bifurcation — find the split point k
    else {
        for (int k = i+1; k < j; k++) {
            if (dp[i*L + j] == dp[i*L + k] + dp[(k+1)*L + j]) {
                traceback_recursive(i,   k, seq, dp, L, structure);
                traceback_recursive(k+1, j, seq, dp, L, structure);
                break;
            }
        }
    }
}

// ─── traceback (public method) ───────────────────────────────────────────────

void GpuNussinov::traceback(DP_PATH& dp_path) {
    for (int s = 0; s < numSeqs; s++) {
        int N = seqs[s].seq.size();

        // extract this sequence's DP table from the flat array
        // dp_path layout: seq0's N×N table, then seq1's, etc.
        // but all tables use longestLen as stride
        DP_PATH seq_dp(dp_path.begin() + s * longestLen * longestLen,
                       dp_path.begin() + (s+1) * longestLen * longestLen);

        // initialize structure to all dots
        std::string structure(N, '.');

        // run traceback for this sequence
        traceback_recursive(0, N-1, seqs[s].seq, seq_dp, longestLen, structure);

        // store results
        seqs[s].structure = structure;
        seqs[s].score     = dp_path[s * longestLen * longestLen + 0 * longestLen + (N-1)];
    }
}

// ─── writeStructures ─────────────────────────────────────────────────────────

void GpuNussinov::writeStructures(std::string fileName, bool append) {
    std::ios_base::openmode mode = append ? (std::ios::out | std::ios::app)
                                          : (std::ios::out);
    std::ofstream outFile(fileName, mode);

    if (!outFile.is_open()) {
        fprintf(stderr, "ERROR: cannot open output file: %s\n", fileName.c_str());
        exit(1);
    }

    for (int s = 0; s < numSeqs; s++) {
        outFile << ">" << seqs[s].name << "\n";
        outFile << seqs[s].seq       << "\n";
        outFile << seqs[s].structure << "\n";
        outFile << seqs[s].score     << "\n";
        outFile << "\n";
    }

    outFile.close();
}

// ─── clearAndReset ───────────────────────────────────────────────────────────

void GpuNussinov::clearAndReset() {
    cudaFree(d_seqs);
    cudaFree(d_dp);
    cudaFree(d_seqLen);
    cudaFree(d_structures);

    d_seqs = nullptr;
    d_dp = nullptr;
    d_seqLen = nullptr;
    d_structures = nullptr;

    seqs.clear();
    longestLen = 0;
    numSeqs    = 0;
}

// ─── runNussinov ─────────────────────────────────────────────────────────────

void GpuNussinov::runNussinov() {
    int threadsPerBlock = 256;

    allocateMem();
    transferSequencesToDevice();

    for (int d = 2; d < longestLen; d++) {
        int cellsPerDiag = longestLen - d;
        int blocksPerSeq = (cellsPerDiag + threadsPerBlock - 1) / threadsPerBlock;

        dim3 grid(blocksPerSeq, numSeqs, 1);
        dim3 block(threadsPerBlock, 1, 1);

        nussinov_kernel<<<grid, block>>>(d_dp, d_seqLen, d_seqs, longestLen, d);
    }

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU Kernel Error: %s (%s)\n",
                cudaGetErrorString(err), cudaGetErrorName(err)); exit(1);
    }

    DP_PATH dp_matrix = transferDPMatrixToHost();
    traceback(dp_matrix);
}