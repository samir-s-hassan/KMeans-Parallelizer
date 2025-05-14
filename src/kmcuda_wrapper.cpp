#include <sstream>        // For stringstream to parse input lines
#include <iostream>       // For standard input/output operations
#include <vector>         // For dynamic arrays
#include <string>         // For string handling
#include <kmcuda.h>       // KM-CUDA library header for GPU-based KMeans
#include <chrono>         // For timing execution
#include <unistd.h>       // For POSIX I/O (pipe, dup, read, close)
#include <fcntl.h>        // For file control options
#include <cstdio>         // For C I/O functions
#include <cstring>        // For C string functions like strtok

using namespace std;

// === Global variables shared between main and helper ===
int total_points, total_values, K;           // Dataset dimensions and cluster count
unsigned long long seed;                     // Random seed for reproducibility
float tolerance = 1e-4f;                     // Convergence tolerance
float yinyang_t = 0.1f;                      // Threshold for Yinyang variant (optional)
vector<float> features;                      // Flat array of point features
vector<float> centroids;                     // Flat array of centroid coordinates
vector<unsigned int> assignments;            // Stores cluster assignment per point

// === Function to execute KM-CUDA and count its iterations from its stdout ===
int count_kmcuda_iterations(KMCUDAResult &out_result)
{
    int pipefd[2];             // Create a pipe to capture stdout
    pipe(pipefd);              // Initialize pipe (pipefd[0] is read, [1] is write)

    int saved_stdout = dup(STDOUT_FILENO);   // Duplicate current stdout to restore later
    dup2(pipefd[1], STDOUT_FILENO);          // Redirect stdout to pipe write end
    close(pipefd[1]);                        // Close duplicate write end (not needed anymore)

    // === Execute KM-CUDA clustering on GPU ===
    out_result = kmeans_cuda(
        kmcudaInitMethodRandom,        // Use random initialization for centroids
        nullptr,                       // No pre-initialized centroids
        tolerance,                     // Convergence tolerance
        yinyang_t,                     // Yinyang threshold (used internally)
        kmcudaDistanceMetricL2,        // Use L2 (Euclidean) distance
        total_points,                  // Number of input points
        static_cast<uint16_t>(total_values), // Number of features per point
        K,                             // Number of clusters
        seed,                          // Random seed
        0, -1, 0,                      // GPU/stream/verbosity options (default or disabled)
        1,                             // Verbosity ON to get log lines like "iteration X"
        features.data(),               // Input feature array
        centroids.data(),              // Output centroids array
        assignments.data(),            // Output point-to-cluster assignment array
        nullptr);                      // No optional score output

    fflush(stdout);                    // Flush any remaining stdout to pipe
    dup2(saved_stdout, STDOUT_FILENO); // Restore original stdout
    close(saved_stdout);               // Close the saved duplicate

    // === Read output from pipe ===
    char buffer[32768];                             // Buffer for captured stdout
    ssize_t count = read(pipefd[0], buffer, sizeof(buffer) - 1); // Read from pipe
    close(pipefd[0]);                               // Close read end

    buffer[count] = '\0';                           // Null-terminate buffer
    int iterations = 0;                             // Counter for KM-CUDA iterations

    char *line = strtok(buffer, "\n");              // Split output into lines
    while (line)
    {
        cout << line << endl;                       // Print each KM-CUDA output line
        if (strncmp(line, "iteration ", 10) == 0)   // Match lines that start with "iteration "
        {
            iterations++;                           // Increment iteration counter
        }
        line = strtok(nullptr, "\n");               // Move to next line
    }

    return iterations;                              // Return how many iterations KM-CUDA ran
}

int main()
{
    int max_iterations, has_name;                   // Read dataset config
    cin >> total_points >> total_values >> K >> max_iterations >> has_name;

    seed = 10;                                       // Set fixed seed for reproducibility

    // Resize vectors to hold input data and results
    features.resize(total_points * total_values);   // Flat vector for all features
    assignments.resize(total_points);               // One cluster assignment per point
    centroids.resize(K * total_values);             // K centroids with `total_values` features

    string dummy;                                   // Used to skip name field if present
    for (int i = 0; i < total_points; ++i)
    {
        string line;
        getline(cin >> ws, line);                   // Read entire input line, skip leading spaces
        stringstream ss(line);                      // Create string stream from line
        string token;

        // Read each feature value up to total_values
        for (int j = 0; j < total_values && getline(ss, token, ','); ++j)
        {
            features[i * total_values + j] = stof(token); // Convert string to float and store
        }

        if (has_name && getline(ss, token, ','))    // Skip name field if present
        {
            dummy = token;
        }
    }

    KMCUDAResult result;                            // Will hold return status
    auto begin = chrono::high_resolution_clock::now(); // Start timer

    // Run KM-CUDA and count iterations
    int actual_iterations = count_kmcuda_iterations(result);

    auto end = chrono::high_resolution_clock::now();    // End timer

    if (result != kmcudaSuccess)                    // Error handling
    {
        cerr << "KMCUDA failed with error code: " << result << endl;
        return 1;
    }

    // === Print results ===
    cout << "Break in iteration " << actual_iterations << "\n\n";
    for (int i = 0; i < K; ++i)
    {
        cout << "Cluster " << i + 1 << "\nCluster values: ";
        for (int j = 0; j < total_values; ++j)
            cout << centroids[i * total_values + j] << " ";
        cout << "\n\n";
    }

    // === Compute and print timing metrics ===
    auto total_us = chrono::duration_cast<chrono::microseconds>(end - begin).count();

    // KM-CUDA runs as one unit, so Phase 1 is mocked as zero
    long long phase1_us = 0;
    long long phase2_us = total_us;

    double avg_time_per_iter = (actual_iterations > 0)
        ? static_cast<double>(phase2_us) / actual_iterations
        : 0.0;

    double throughput = (double)(total_points * actual_iterations) / (phase2_us / 1e6); // points/sec
    double latency = (double)phase2_us / (total_points * actual_iterations);            // µs/point

    // === Match output format for comparisons ===
    cout << "TOTAL EXECUTION TIME = " << total_us << " µs\n";
    cout << "TIME PHASE 1 = " << phase1_us << " µs\n";
    cout << "TIME PHASE 2 = " << phase2_us << " µs\n";
    cout << "KMCUDA, AVERAGE TIME PER ITERATION = " << avg_time_per_iter << " µs\n";
    cout << "PHASE 2 THROUGHPUT = " << throughput << " points per second\n";
    cout << "PHASE 2 LATENCY = " << latency << " µs per point\n";

    return 0;
}
