#pragma once
#include<stdint.h>
#include <vector>
#include <string>
#include <assert.h>
#include "timer.hpp"

void printGpuProperties();

using DP_PATH = std::vector<int32_t>;

struct RNASequence{
    int id;
    std::string name;
    std::string seq;
    std::string structure;
    int32_t score; 
    RNASequence(int _id, std::string _name, std::string _seq): id(_id), name(_name), seq(_seq), score(0) {};
};

struct GpuNussinov{
   // CPU-side
    std::vector<RNASequence> seqs;
    int32_t longestLen;   // longest sequence in batch — sizes the DP table
    int32_t numSeqs;      // number of sequences in current batch

    // GPU-side (d_ prefix = device memory)
    char*    d_seqs;        // packed sequences flat array
    int32_t* d_dp;          // DP tables — one N×N table per sequence
    int32_t* d_seqLen;      // length of each sequence
    char*    d_structures;  // dot-bracket output strings

    // Pipeline methods
    void runNussinov();                                     // main entry point
    void runNussinovV2();
    void runNussinovV3();
    void runNussinovSequential(); 
    void allocateMem();                                     // cudaMalloc all d_ pointers
    void transferSequencesToDevice();                        // cudaMemcpy seqs CPU → GPU
    DP_PATH transferDPMatrixToHost();                             // cudaMemcpy DP table GPU → CPU
    void traceback(DP_PATH& dp_table);                     // CPU traceback → fills structure field
    void writeStructures(std::string fileName, bool append);// write dot-bracket output
    void clearAndReset();                                   // cudaFree + clear seqs vector

    GpuNussinov() : longestLen(0), numSeqs(0),
                    d_seqs(nullptr), d_dp(nullptr),
                    d_seqLen(nullptr), d_structures(nullptr) {}
};