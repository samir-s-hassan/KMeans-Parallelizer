#!/bin/bash

# === CONFIGURATION ===
DATASET_DIR="datasets"
DEFAULT_DATASET="1.txt"
CSV_OUTPUT_DIR="cluster_results"
EXECUTABLE_DIR="executables"
OUTPUT_FILE="results.txt"

# === CLEAN OLD OUTPUT ===
[ -d "$CSV_OUTPUT_DIR" ] && rm -f "$CSV_OUTPUT_DIR"/*
mkdir -p "$EXECUTABLE_DIR"

# === IMPLEMENTATION MAP ===
declare -A IMPLEMENTATIONS=(
    [s]="src/serial.cpp serial"
    [p]="src/parallel.cpp parallel"
    [a]="src/a-parallel.cpp a-parallel"
    [b]="src/b-parallel.cpp b-parallel"
    [k]="src/kmcuda_wrapper.cpp kmcuda"
    [r]="src/[ARCHIVED]kmc.cpp [ARCHIVED]kmc"
)

# === ENVIRONMENT ===
source /etc/profile.d/modules.sh
module load gcc-11.2.0
cd oneapi-tbb-2022.0.0/env || { echo "❌ TBB not found."; exit 1; }
source vars.sh
cd ../..

# === PARSE ARGS ===
SELECTED_IMPLEMENTATIONS=()
DATASET=""
for ARG in "$@"; do
    if [[ -n ${IMPLEMENTATIONS[$ARG]} ]]; then
        SELECTED_IMPLEMENTATIONS+=("$ARG")
    else
        DATASET="$ARG"
    fi
done

[ -z "$DATASET" ] && DATASET="$DATASET_DIR/$DEFAULT_DATASET" || DATASET="$DATASET_DIR/$DATASET"
if [ ! -f "$DATASET" ]; then
    echo "❌ Dataset not found: $DATASET"
    exit 1
fi

# === START LOGGING ===
echo "📊 Using dataset: $DATASET"
echo "Running K-Means Implementations on $DATASET" > "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

[ ${#SELECTED_IMPLEMENTATIONS[@]} -eq 0 ] && SELECTED_IMPLEMENTATIONS=("s" "p" "a" "b" "k")

# === COMPILE & RUN ===
for IMPL in "${SELECTED_IMPLEMENTATIONS[@]}"; do
    read -r SOURCE_FILE EXECUTABLE <<< "${IMPLEMENTATIONS[$IMPL]}"
    EXECUTABLE_PATH="./$EXECUTABLE_DIR/$EXECUTABLE"

    echo "🔧 Compiling $EXECUTABLE..."
    if [[ "$IMPL" == "k" || "$IMPL" == "r" ]]; then
        # KM-CUDA settings
        KMCUDA_DIR="kmcuda"
        KMCUDA_INCLUDE="$KMCUDA_DIR/src"
        KMCUDA_LIB="$KMCUDA_DIR/build"
        g++ -std=c++11 -O3 \
            -I"$KMCUDA_INCLUDE" \
            -L"$KMCUDA_LIB" -Wl,-rpath="$KMCUDA_LIB" \
            -lKMCUDA "$SOURCE_FILE" -o "$EXECUTABLE_PATH"
    elif [[ "$IMPL" == "p" || "$IMPL" == "a" || "$IMPL" == "b" ]]; then
        g++ -std=c++11 -O3 -march=native \
            -I$TBBROOT/include \
            -L$TBBROOT/lib/intel64/gcc4.8 \
            -ltbb -ltbbmalloc -ltbbmalloc_proxy \
            "$SOURCE_FILE" -o "$EXECUTABLE_PATH"
    else
        g++ -std=c++11 -O3 -march=native "$SOURCE_FILE" -o "$EXECUTABLE_PATH"
    fi

    echo "🚀 Running $EXECUTABLE on $DATASET"
    echo "" >> "$OUTPUT_FILE"
    echo "===== Running $EXECUTABLE on $DATASET =====" >> "$OUTPUT_FILE"
    cat "$DATASET" | "$EXECUTABLE_PATH" >> "$OUTPUT_FILE" 2>&1
    echo "$EXECUTABLE Execution Completed!" >> "$OUTPUT_FILE"
    echo ""
done

# === CLEANUP ===
rm -rf "$EXECUTABLE_DIR"
echo "✅ All executions completed. Results saved to $OUTPUT_FILE"
