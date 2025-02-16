#!/bin/bash

# Set the source file name and output executable
SOURCE_FILE="kmeans-serial.cpp"
EXECUTABLE="kmeans"
DATASET_DIR="datasets"  # Default dataset directory

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

# Navigate back to the root directory (two levels up)
cd ../..

# Check if the source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file '$SOURCE_FILE' not found! Please place it in the current directory."
    exit 1
fi

# Compile with -O3 optimization and link TBB
g++ -std=c++11 -O3 "$SOURCE_FILE" -o "$EXECUTABLE" -ltbb
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Check if dataset filename is provided as an argument
if [ -z "$1" ]; then
    echo "Error: No dataset file provided!"
    echo "Usage: ./setup_kmeans.sh sample.txt"
    exit 1
fi

DATASET="$DATASET_DIR/$1"  # Automatically prepend "datasets/"

# Check if the dataset file exists
if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset file '$DATASET' not found in '$DATASET_DIR'!"
    exit 1
fi

echo "===== Running K-Means on Dataset: $DATASET ====="
echo ""

# Run K-Means with the provided dataset
cat "$DATASET" | ./"$EXECUTABLE"

echo "===== K-Means Execution Completed! ====="
echo ""
