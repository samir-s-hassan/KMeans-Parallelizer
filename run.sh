#!/bin/bash

# Set the dataset directory and default dataset
DATASET_DIR="datasets"
DEFAULT_DATASET="1.txt"

# Define implementation mappings
declare -A IMPLEMENTATIONS
IMPLEMENTATIONS=(
    [s]="kmeans-serial.cpp kmeans-serial"
    [f]="kmeans-fast-serial.cpp kmeans-fast-serial"
    [c]="kmeans-concurrent.cpp kmeans-concurrent"
    [p]="kmeans-parallel.cpp kmeans-parallel"
)

# Load the correct GCC module
module load gcc-11.2.0

# Verify GCC version
GCC_VERSION=$(gcc --version | head -n 1)
if [[ ! "$GCC_VERSION" =~ "11.2.0" ]]; then
    echo "Error: Failed to switch to GCC 11.2.0. Current version is: $GCC_VERSION"
    exit 1
fi

# Navigate to the TBB environment setup directory
cd oneapi-tbb-2022.0.0/env || { echo "Error: TBB directory not found"; exit 1; }

# Source the environment variables
source vars.sh

# Navigate back to the root directory
cd ../..

# Parse arguments to determine implementations and dataset
SELECTED_IMPLEMENTATIONS=()
DATASET=""
for ARG in "$@"; do
    if [[ -n ${IMPLEMENTATIONS[$ARG]} ]]; then
        SELECTED_IMPLEMENTATIONS+=("$ARG")
    else
        DATASET="$ARG"
    fi
done

# Use default dataset if none was provided
if [ -z "$DATASET" ]; then
    echo "No dataset file provided. Using default dataset: $DEFAULT_DATASET"
    echo ""
    DATASET="$DATASET_DIR/$DEFAULT_DATASET"
else
    DATASET="$DATASET_DIR/$DATASET"
    echo "Using dataset: $DATASET"
    echo ""
fi

# Check if the dataset file exists
if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset file '$DATASET' not found in '$DATASET_DIR'!"
    exit 1
fi

# Output file for results (overwrite file at the start)
OUTPUT_FILE="results.txt"
echo "Running K-Means Implementations on $DATASET" > "$OUTPUT_FILE"  # Use '>' instead of '>>'
echo "" >> "$OUTPUT_FILE"

# If no valid implementations were provided, default to serial (s)
if [ ${#SELECTED_IMPLEMENTATIONS[@]} -eq 0 ]; then
    SELECTED_IMPLEMENTATIONS=("s")
fi

# Loop through selected implementations
for IMPL in "${SELECTED_IMPLEMENTATIONS[@]}"; do
    read -r SOURCE_FILE EXECUTABLE <<< "${IMPLEMENTATIONS[$IMPL]}"

    # echo "===== Compiling $SOURCE_FILE ====="
    
    # Check if source file exists
    if [ ! -f "$SOURCE_FILE" ]; then
        echo "Error: Source file '$SOURCE_FILE' not found! Skipping..."
        echo "Error: Source file '$SOURCE_FILE' not found! Skipping..." >> "$OUTPUT_FILE"
        continue
    fi

    # Compile the implementation
    g++ -std=c++11 -O3 "$SOURCE_FILE" -o "$EXECUTABLE" -ltbb
    if [ $? -ne 0 ]; then
        echo "Compilation failed for $SOURCE_FILE! Skipping..."
        echo "Compilation failed for $SOURCE_FILE! Skipping..." >> "$OUTPUT_FILE"
        continue
    fi

    # echo "===== Running $EXECUTABLE on $DATASET ====="
    echo "===== Running $EXECUTABLE on $DATASET =====" >> "$OUTPUT_FILE"

    # Run K-Means and append results to output file (without terminal output)
    cat "$DATASET" | ./$EXECUTABLE >> "$OUTPUT_FILE" 2>&1

    echo "===== $EXECUTABLE Execution Completed! ====="
    # echo "===== $EXECUTABLE Execution Completed! =====" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

# ========= PARSING RESULTS & DISPLAYING SUMMARY =========
echo -e "\n======== Summary of Results ========"

# Read the results file line by line
IMPLEMENTATION=""
AVERAGE_TIME=""
CLUSTER_VALUES=""
ITERATIONS=""
TIME_PHASE_2=""

while IFS= read -r line; do
    # Detect when a new implementation is being processed
    if [[ "$line" =~ ^=====.*Running.*on.*$ ]]; then
        # If we already have data, print it before starting the next one
        if [[ -n "$IMPLEMENTATION" && -n "$AVERAGE_TIME" && -n "$CLUSTER_VALUES" && -n "$ITERATIONS" && -n "$TIME_PHASE_2" ]]; then
            echo -e "$IMPLEMENTATION:\n  - Time Phase 2: $TIME_PHASE_2\n  - Iterations: $ITERATIONS\n  - Average Time per Iteration: $AVERAGE_TIME\n  - Final Cluster Values: $CLUSTER_VALUES\n"
        fi
        IMPLEMENTATION=$(echo "$line" | awk '{print $3}')
        AVERAGE_TIME=""
        CLUSTER_VALUES=""
        ITERATIONS=""
        TIME_PHASE_2=""
    
    # Extract Time Phase 2
    elif [[ "$line" =~ TIME\ PHASE\ 2\ = ]]; then
        TIME_PHASE_2=$(echo "$line" | awk -F' = ' '{print $2}')
    
    # Extract Iteration Count (Break in iteration X)
    elif [[ "$line" =~ Break\ in\ iteration ]]; then
        ITERATIONS=$(echo "$line" | awk '{print $4}')
    
    # Extract Average Time Per Iteration
    elif [[ "$line" =~ AVERAGE\ TIME\ PER\ ITERATION\ = ]]; then
        AVERAGE_TIME=$(echo "$line" | awk -F' = ' '{print $2}')
    
    # Extract Final Cluster Values (last occurrence)
    elif [[ "$line" =~ Cluster\ values: ]]; then
        CLUSTER_VALUES=$(echo "$line" | awk -F': ' '{print $2}')
    fi
done < "$OUTPUT_FILE"

# Print the last implementation if it has values
if [[ -n "$IMPLEMENTATION" && -n "$AVERAGE_TIME" && -n "$CLUSTER_VALUES" && -n "$ITERATIONS" && -n "$TIME_PHASE_2" ]]; then
    echo -e "$IMPLEMENTATION:\n  - Time Phase 2: $TIME_PHASE_2\n  - Iterations: $ITERATIONS\n  - Average Time per Iteration: $AVERAGE_TIME\n  - Final Cluster Values: $CLUSTER_VALUES\n"
fi

echo "Full results saved in $(pwd)/$OUTPUT_FILE"
echo " "
