#!/bin/bash

# 1. Define paths
SOURCE_FILE="./src/nussinov_cpu.cpp"
EXE_FILE="./src/nussinov_baseline.exe"
INPUT_DIR="./data/baseline_bins"
OUTPUT_DIR="./data/reference_structures"

echo "--- Starting Nussinov Baseline Setup ---"

# 2. Create directories if they don't exist
echo "Checking directories..."
mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# 3. Compile the code
# -O3 tells the compiler to optimize for speed
echo "Compiling $SOURCE_FILE..."
g++ -O3 "$SOURCE_FILE" -o "$EXE_FILE"

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "----------------------------------------"
    
    # 4. Run the program
    echo "Running baseline..."
    "$EXE_FILE"
    
    echo "----------------------------------------"
    echo "Done! Check $OUTPUT_DIR for results."
else
    echo "ERROR: Compilation failed. Please check your C++ code."
fi