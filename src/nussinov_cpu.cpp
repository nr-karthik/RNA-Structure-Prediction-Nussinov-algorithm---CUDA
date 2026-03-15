/*
 Nussinov Algorithm - CPU Reference Implementation
 Output: Standard dot-bracket notation for correctness validation
*/

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>

#define MAX_N 1001

using namespace std;

int n;
string A;
int dp[MAX_N][MAX_N];

bool complementary(char X, char Y)
{
    return ((X == 'A' && Y == 'U') || (X == 'U' && Y == 'A') ||
            (X == 'C' && Y == 'G') || (X == 'G' && Y == 'C') ||
            (X == 'G' && Y == 'U') || (X == 'U' && Y == 'G'));
}

int nussinov(int i, int j)
{
    if (dp[i][j] != -1) return dp[i][j];

    if (i >= j - 1)
    {
        dp[i][j] = 0;
        return 0;
    }

    int ret = 0;
    ret = max(ret, nussinov(i + 1, j));
    ret = max(ret, nussinov(i, j - 1));
    if (complementary(A[i], A[j]))
        ret = max(ret, nussinov(i + 1, j - 1) + 1);
    for (int k = i + 1; k < j; k++)
        ret = max(ret, nussinov(i, k) + nussinov(k + 1, j));

    dp[i][j] = ret;
    return ret;
}

inline int nussinov()
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dp[i][j] = -1;

    return nussinov(0, n - 1);
}

// Standard dot-bracket traceback
void traceback(int i, int j, string &structure)
{
    if (i >= j) return;

    if (dp[i][j] == dp[i + 1][j])
    {
        structure[i] = '.';
        traceback(i + 1, j, structure);
    }
    else if (dp[i][j] == dp[i][j - 1])
    {
        structure[j] = '.';
        traceback(i, j - 1, structure);
    }
    else if (complementary(A[i], A[j]) && dp[i][j] == dp[i + 1][j - 1] + 1)
    {
        structure[i] = '(';
        structure[j] = ')';
        traceback(i + 1, j - 1, structure);
    }
    else
    {
        for (int k = i + 1; k < j; k++)
        {
            if (dp[i][j] == dp[i][k] + dp[k + 1][j])
            {
                traceback(i, k, structure);
                traceback(k + 1, j, structure);
                break;
            }
        }
    }
}

int main(){   
    
    ifstream infile("data/baseline_bins/seq_512.fa");
    ofstream myFile("data/reference_structures/ref_seq_512.txt");

    if(!infile.is_open()){
        printf("Error: could not open file\n");
        return 1;
    }

    if(!myFile.is_open()){
        printf("Error: could not create a new file to store the reference structures.\n");
        return 1;
    }

    string line, header, sequence;
    int sequence_count = 0;
    int max_seqs = 1;

    while(getline(infile, line)){
        if(line.empty()) continue;

        if(line[0] == '>'){
            if(!sequence.empty()){
                A = sequence;
                n = A.length();

                int score = nussinov();
                string structure(n, '.');
                traceback(0, n-1, structure);

                printf("Header: %s\n", header.c_str());
                printf("Sequence: %s\n", A.c_str());
                printf("Structure: %s\n", structure.c_str());
                printf("Score: %d\n\n", score);
                
                if(myFile.is_open()){
                    myFile << header << endl;
                    myFile << A << endl;
                    myFile << structure << endl;
                    myFile << score << endl << endl;
                }
                sequence_count++;
                if(sequence_count >= max_seqs) break;
            }
            header = line;
            sequence = "";
        }else{
            sequence += line;
        }
    }

    if (sequence_count < max_seqs && !sequence.empty()) {
        A = sequence;
        n = A.length();
        int score = nussinov();
        string structure(n, '.');
        traceback(0, n - 1, structure);

        myFile << header << endl;
        myFile << A << endl;
        myFile << structure << endl;
        myFile << score << endl << endl;
        sequence_count++;
    }

    infile.close();
    myFile.close();
    printf("Processed %d sequences. \n", sequence_count);
    return 0;
}