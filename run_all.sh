#!/bin/bash

# Set the dataset directory and default dataset
DATASET_DIR="datasets"
DEFAULT_DATASET="1.txt"

# List of implementations (source file -> executable name)
IMPLEMENTATIONS=(
    "kmeans-concurrent.cpp kmeans-concurrent"
    "kmeans-serial-fast.cpp kmeans-serial-fast"
    "kmeans-parallel.cpp kmeans-parallel"
    "kmeans-serial.cpp kmeans"
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

# Check if dataset filename is provided as an argument, else use default
if [ -z "$1" ]; then
    echo "No dataset file provided. Using default dataset: $DEFAULT_DATASET"
    DATASET="$DATASET_DIR/$DEFAULT_DATASET"
else
    DATASET="$DATASET_DIR/$1"
fi

# Check if the dataset file exists
if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset file '$DATASET' not found in '$DATASET_DIR'!"
    exit 1
fi

# Output file for results
OUTPUT_FILE="kmeans_results.txt"
echo "Running K-Means Implementations on dataset: $DATASET" > "$OUTPUT_FILE"

# Loop through each implementation
for IMPL in "${IMPLEMENTATIONS[@]}"; do
    # Split the string into source file and executable name
    read -r SOURCE_FILE EXECUTABLE <<< "$IMPL"

    echo ""
    echo "Compiling $SOURCE_FILE..."
    
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

    echo "===== Running $EXECUTABLE on: $DATASET ====="
    echo "===== Running $EXECUTABLE on: $DATASET =====" >> "$OUTPUT_FILE"

    # Run K-Means and append results to output file
    cat "$DATASET" | ./"$EXECUTABLE" >> "$OUTPUT_FILE" 2>&1

    echo "===== $EXECUTABLE Execution Completed! ====="
    echo "===== $EXECUTABLE Execution Completed! =====" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo ""
done

echo "All implementations completed. Results saved in $OUTPUT_FILE."
