#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>

using namespace std;

bool complementary(char X, char Y) {
    return ((X=='A'&&Y=='U')||(X=='U'&&Y=='A')||
            (X=='C'&&Y=='G')||(X=='G'&&Y=='C')||
            (X=='G'&&Y=='U')||(X=='U'&&Y=='G'));
}

int nussinov_iterative(const string& seq) {
    int n = seq.size();
    vector<vector<int>> dp(n, vector<int>(n, 0));
    for (int d = 2; d < n; d++) {
        for (int i = 0; i < n - d; i++) {
            int j = i + d;
            int result = 0;
            if (complementary(seq[i], seq[j]))
                result = max(result, dp[i+1][j-1] + 1);
            for (int k = i; k < j; k++)
                result = max(result, dp[i][k] + dp[k+1][j]);
            dp[i][j] = result;
        }
    }
    return dp[0][n-1];
}

int main(int argc, char** argv) {
    string inputFile = "data/baseline_bins/seq_256.fa";
    int max_seqs = 1000;
    int num_threads = 4;

    if (argc > 1) inputFile   = argv[1];
    if (argc > 2) max_seqs    = atoi(argv[2]);
    if (argc > 3) num_threads = atoi(argv[3]);

    // Read all sequences first
    ifstream infile(inputFile);
    if (!infile.is_open()) { printf("Error opening %s\n", inputFile.c_str()); return 1; }

    vector<string> sequences;
    string line, sequence;
    while (getline(infile, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            if (!sequence.empty()) {
                sequences.push_back(sequence);
                if ((int)sequences.size() >= max_seqs) break;
            }
            sequence = "";
        } else {
            sequence += line;
        }
    }
    if ((int)sequences.size() < max_seqs && !sequence.empty())
        sequences.push_back(sequence);

    printf("Processing %zu sequences with %d threads...\n", sequences.size(), num_threads);
    fflush(stdout);

    auto start = chrono::high_resolution_clock::now();

    // Parallel loop over sequences
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int s = 0; s < (int)sequences.size(); s++) {
        nussinov_iterative(sequences[s]);
    }

    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

    printf("CPU DP-only time: %.2f ms using %d threads\n", elapsed * 1000, num_threads);
    printf("CPU processed %zu sequences in %.2f seconds using %d threads\n",
           sequences.size(), elapsed, num_threads);
    return 0;
}
