#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <kmcuda.h>
#include <chrono>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstring>

using namespace std;

// Global variables to share between main and counting function
int total_points, total_values, K;
unsigned long long seed;
float tolerance = 1e-4f;
float yinyang_t = 0.1f;
vector<float> features;
vector<float> centroids;
vector<unsigned int> assignments;

int count_kmcuda_iterations(KMCUDAResult &out_result)
{
    int pipefd[2];
    pipe(pipefd);

    // Backup stdout
    int saved_stdout = dup(STDOUT_FILENO);
    dup2(pipefd[1], STDOUT_FILENO); // Redirect stdout
    close(pipefd[1]);

    // Run KMCUDA
    out_result = kmeans_cuda(
        kmcudaInitMethodRandom,
        nullptr,
        tolerance,
        yinyang_t,
        kmcudaDistanceMetricL2,
        total_points,
        static_cast<uint16_t>(total_values),
        K,
        seed,
        0,
        -1,
        0,
        1, // verbosity
        features.data(),
        centroids.data(),
        assignments.data(),
        nullptr);

    fflush(stdout);
    dup2(saved_stdout, STDOUT_FILENO); // Restore stdout
    close(saved_stdout);

    char buffer[32768];
    ssize_t count = read(pipefd[0], buffer, sizeof(buffer) - 1);
    close(pipefd[0]);

    buffer[count] = '\0';
    int iterations = 0;

    char *line = strtok(buffer, "\n");
    while (line)
    {
        // KEEP THIS FOR DRY BEANS, COMMENT FOR BIG DATASETS
        cout << line << endl; // <-- Echo back KM-CUDA's log output
        if (strncmp(line, "iteration ", 10) == 0)
        {
            iterations++;
        }
        line = strtok(nullptr, "\n");
    }

    return iterations;
}

int main()
{
    int max_iterations, has_name;
    cin >> total_points >> total_values >> K >> max_iterations >> has_name;

    seed = 10;

    features.resize(total_points * total_values);
    assignments.resize(total_points);
    centroids.resize(K * total_values);

    string dummy;
    for (int i = 0; i < total_points; ++i)
    {
        string line;
        getline(cin >> ws, line); // Read full line, skip leading whitespace
        stringstream ss(line);
        string token;

        for (int j = 0; j < total_values && getline(ss, token, ','); ++j)
        {
            features[i * total_values + j] = stof(token); // Convert token to float
        }

        if (has_name && getline(ss, token, ',')) // Read name if present
        {
            dummy = token; // Not used, but captured if needed
        }
    }

    KMCUDAResult result;
    auto begin = chrono::high_resolution_clock::now();
    int actual_iterations = count_kmcuda_iterations(result);
    auto end = chrono::high_resolution_clock::now();

    if (result != kmcudaSuccess)
    {
        cerr << "KMCUDA failed with error code: " << result << endl;
        return 1;
    }

    // === Print Clustering Info ===
    cout << "Break in iteration " << actual_iterations << "\n\n";
    for (int i = 0; i < K; ++i)
    {
        cout << "Cluster " << i + 1 << "\nCluster values: ";
        for (int j = 0; j < total_values; ++j)
            cout << centroids[i * total_values + j] << " ";
        cout << "\n\n";
    }

    // === Print Timing Metrics ===
    auto total_us = chrono::duration_cast<chrono::microseconds>(end - begin).count();

    // You can’t split KM-CUDA into phases, but fake Phase 1 = 0
    long long phase1_us = 0;
    long long phase2_us = total_us;

    double avg_time_per_iter = (actual_iterations > 0)
                                   ? static_cast<double>(phase2_us) / actual_iterations
                                   : 0.0;

    double throughput = (double)(total_points * actual_iterations) / (phase2_us / 1e6);
    double latency = (double)phase2_us / (total_points * actual_iterations);

    // === Match Output Format ===
    cout << "TOTAL EXECUTION TIME = " << total_us << " µs\n";
    cout << "TIME PHASE 1 = " << phase1_us << " µs\n";
    cout << "TIME PHASE 2 = " << phase2_us << " µs\n";
    cout << "KMCUDA, AVERAGE TIME PER ITERATION = " << avg_time_per_iter << " µs\n";
    cout << "PHASE 2 THROUGHPUT = " << throughput << " points per second\n";
    cout << "PHASE 2 LATENCY = " << latency << " µs per point\n";

    return 0;
}
