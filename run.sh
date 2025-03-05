#!/bin/bash

# Set the dataset directory and default dataset
DATASET_DIR="datasets"
DEFAULT_DATASET="1.txt"

# Remove all files inside cluster_results without deleting the folder
CSV_OUTPUT_DIR="cluster_results"
if [ -d "$CSV_OUTPUT_DIR" ]; then
    rm -f "$CSV_OUTPUT_DIR"/*
fi


# Define implementation mappings (Pointing to src/)
declare -A IMPLEMENTATIONS
IMPLEMENTATIONS=(
    [s]="src/serial.cpp serial"
    [f]="src/fast-serial.cpp fast-serial"
    # [c]="src/concurrent.cpp concurrent"
    [p]="src/parallel.cpp parallel"
    [n]="src/na-serial.cpp na-serial"
    [l]="src/lightning-serial.cpp lightning-serial"
    [a]="src/a-parallel.cpp a-parallel"
    [b]="src/b-parallel.cpp b-parallel"
    [u]="src/usion-parallel.cpp usion-parallel"
)

# Initialize the module system
source /etc/profile.d/modules.sh  # This is usually required on many systems

# Load GCC 11.2.0
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
    DATASET="$DATASET_DIR/$DEFAULT_DATASET"
else
    DATASET="$DATASET_DIR/$DATASET"
    echo "Using dataset: $DATASET"
fi

# Check if the dataset file exists
if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset file '$DATASET' not found in '$DATASET_DIR'!"
    exit 1
fi

# Output file for results (overwrite file at the start)
OUTPUT_FILE="results.txt"
echo "Running K-Means Implementations on $DATASET" > "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# If no valid implementations were provided, default to serial (s)
if [ ${#SELECTED_IMPLEMENTATIONS[@]} -eq 0 ]; then
    SELECTED_IMPLEMENTATIONS=("s")
fi

# Define directories
EXECUTABLE_DIR="executables"

# Ensure executables directory exists
mkdir -p "$EXECUTABLE_DIR"

# Loop through selected implementations
for IMPL in "${SELECTED_IMPLEMENTATIONS[@]}"; do
    read -r SOURCE_FILE EXECUTABLE <<< "${IMPLEMENTATIONS[$IMPL]}"

    # Define the path for the executable
    EXECUTABLE_PATH="./$EXECUTABLE_DIR/$EXECUTABLE"

    # Check if source file exists in src/
    if [ ! -f "$SOURCE_FILE" ]; then
        echo "Error: Source file '$SOURCE_FILE' not found! Skipping..."
        echo "Error: Source file '$SOURCE_FILE' not found! Skipping..." >> "$OUTPUT_FILE"
        continue
    fi

    # Compile the implementation and place the executable in the folder
    if [[ "$IMPL" == "p" || "$IMPL" == "a" || "$IMPL" == "b" || "$IMPL" == "u" ]]; then
        g++ -std=c++11 -O3 -march=native \
            -I$TBBROOT/include \
            -L$TBBROOT/lib/intel64/gcc4.8 \
            -ltbb -ltbbmalloc -ltbbmalloc_proxy \
            "$SOURCE_FILE" -o "$EXECUTABLE_PATH"
    else
        g++ -std=c++11 -O3 -march=native "$SOURCE_FILE" -o "$EXECUTABLE_PATH"
    fi

    # Run K-Means and append results to output file
    echo "===== Running $EXECUTABLE on $DATASET =====" >> "$OUTPUT_FILE"
    echo "===== Running $EXECUTABLE on $DATASET ====="
    cat "$DATASET" | "$EXECUTABLE_PATH" >> "$OUTPUT_FILE" 2>&1
    echo "$EXECUTABLE Execution Completed!" >> "$OUTPUT_FILE"
    echo "===== $EXECUTABLE Execution Completed! ====="
    echo ""
    echo "" >> "$OUTPUT_FILE"
done

# ========= PARSING RESULTS & DISPLAYING SUMMARY =========
echo -e "======== Summary of Results ========"

# Read the results file line by line
IMPLEMENTATION=""
AVERAGE_TIME=""
CLUSTER_VALUES=""
ITERATIONS=""
TIME_PHASE_2=""
THROUGHPUT=""
LATENCY=""

while IFS= read -r line; do
    # Detect when a new implementation is being processed
    if [[ "$line" =~ ^=====.*Running.*on.*$ ]]; then
        # Print previous implementation details if available
        if [[ -n "$IMPLEMENTATION" && -n "$AVERAGE_TIME" && -n "$CLUSTER_VALUES" && -n "$ITERATIONS" && -n "$TIME_PHASE_2" && -n "$THROUGHPUT" && -n "$LATENCY" ]]; then
            echo -e "$IMPLEMENTATION:\n  - Time Phase 2: $TIME_PHASE_2\n  - Iterations: $ITERATIONS\n  - Average Time per Iteration: $AVERAGE_TIME\n  - Throughput (Phase 2): $THROUGHPUT\n  - Latency (Phase 2): $LATENCY\n  - Final Cluster Values: $CLUSTER_VALUES\n"
        fi
        
        # Reset variables for the new implementation
        IMPLEMENTATION=$(echo "$line" | awk '{print $3}')
        AVERAGE_TIME=""
        CLUSTER_VALUES=""
        ITERATIONS=""
        TIME_PHASE_2=""
        THROUGHPUT=""
        LATENCY=""
    
    # Extract Time Phase 2
    elif [[ "$line" =~ TIME\ PHASE\ 2\ = ]]; then
        TIME_PHASE_2=$(echo "$line" | awk -F' = ' '{print $2}')
    
    # Extract Iteration Count (Break in iteration X)
    elif [[ "$line" =~ Break\ in\ iteration ]]; then
        ITERATIONS=$(echo "$line" | awk '{print $4}')
    
    # Extract Average Time Per Iteration
    elif [[ "$line" =~ AVERAGE\ TIME\ PER\ ITERATION\ = ]]; then
        AVERAGE_TIME=$(echo "$line" | awk -F' = ' '{print $2}')
    
    # Extract Throughput
    elif [[ "$line" =~ THROUGHPUT\ = ]]; then
        THROUGHPUT=$(echo "$line" | awk -F' = ' '{print $2}')
    
    # Extract Latency
    elif [[ "$line" =~ LATENCY\ = ]]; then
        LATENCY=$(echo "$line" | awk -F' = ' '{print $2}')
    
    # Extract Final Cluster Values (last occurrence)
    elif [[ "$line" =~ Cluster\ values: ]]; then
        CLUSTER_VALUES=$(echo "$line" | awk -F': ' '{print $2}')
    fi
done < "$OUTPUT_FILE"

# Print last implementation results if available
if [[ -n "$IMPLEMENTATION" && -n "$AVERAGE_TIME" && -n "$CLUSTER_VALUES" && -n "$ITERATIONS" && -n "$TIME_PHASE_2" && -n "$THROUGHPUT" && -n "$LATENCY" ]]; then
    echo -e "$IMPLEMENTATION:\n  - Time Phase 2: $TIME_PHASE_2\n  - Iterations: $ITERATIONS\n  - Average Time per Iteration: $AVERAGE_TIME\n  - Throughput (Phase 2): $THROUGHPUT\n  - Latency (Phase 2): $LATENCY\n  - Final Cluster Values: $CLUSTER_VALUES\n"
fi

echo "✅ Full results saved in $(pwd)/$OUTPUT_FILE"

# # ========= GENERATE CLUSTER CSV FILES =========
# GEN_CLUSTER_SCRIPT="generate_csv.py"
# CSV_OUTPUT_DIR="cluster_results"

# if [ -f "$GEN_CLUSTER_SCRIPT" ]; then
#     python3 "$GEN_CLUSTER_SCRIPT"
    
#     if [ -d "$CSV_OUTPUT_DIR" ] && [ "$(ls -A "$CSV_OUTPUT_DIR")" ]; then
#         echo ""
#     else
#         echo "⚠️ Warning: CSV files were not generated! Please check '$GEN_CLUSTER_SCRIPT'."
#     fi
# else
#     echo "❌ Error: '$GEN_CLUSTER_SCRIPT' not found! Please make sure the script exists."
# fi

# ========= FINISH =========
rm -rf "$EXECUTABLE_DIR"
