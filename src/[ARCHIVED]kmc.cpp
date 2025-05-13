#include <iostream>
#include <vector>
#include <string>
#include <kmcuda.h>
#include <chrono>

using namespace std;

int main()
{
    int total_points, total_values, K, max_iterations, has_name;
    std::cin >> total_points >> total_values >> K >> max_iterations >> has_name;

    std::vector<float> features(total_points * total_values);
    std::string dummy;
    for (int i = 0; i < total_points; ++i)
    {
        for (int j = 0; j < total_values; ++j)
        {
            std::cin >> features[i * total_values + j];
        }
        if (has_name)
            std::cin >> dummy;
    }

    std::vector<unsigned int> assignments(total_points);
    std::vector<float> centroids(K * total_values);

    float tolerance = 1e-4f;
    float yinyang_t = 0.1f;
    unsigned long long seed = 10;

    auto begin = chrono::high_resolution_clock::now();

    KMCUDAResult result = kmeans_cuda(
        kmcudaInitMethodRandom,              // Initialization method
        nullptr,                             // init_params
        tolerance,                           // Tolerance
        yinyang_t,                           // Yinyang threshold
        kmcudaDistanceMetricL2,              // Distance metric
        total_points,                        // Number of samples
        static_cast<uint16_t>(total_values), // Number of features
        K,                                   // Number of clusters
        seed,                                // Random seed
        0,                                   // Use all available CUDA devices
        -1,                                  // Data on host
        0,                                   // Not using fp16x2
        1,                                   // Verbosity
        features.data(),                     // Input data
        centroids.data(),                    // Output centroids
        assignments.data(),                  // Output assignments
        nullptr                              // Average distance (not computed)
    );

    auto end = chrono::high_resolution_clock::now();

    if (result != kmcudaSuccess)
    {
        std::cerr << "KMCUDA failed with error code: " << result << std::endl;
        return 1;
    }

    auto total_us = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    for (int i = 0; i < K; ++i)
    {
        cout << "Cluster " << i + 1 << "\nCluster values: ";
        for (int j = 0; j < total_values; ++j)
            cout << centroids[i * total_values + j] << " ";
        cout << "\n\n";
    }

    cout << "TOTAL EXECUTION TIME = " << total_us << " µs\n";
    // We are not splitting phases; KM-CUDA is monolithic
    // cout << "TIME PHASE 1 = 0 µs\n";
    // cout << "TIME PHASE 2 = " << total_us << " µs\n";

    // Assume it always runs max_iterations for comparison
    // double avg_time_per_iter = static_cast<double>(total_us) / max_iterations;
    // cout << "KMCUDA, AVERAGE TIME PER ITERATION = " << avg_time_per_iter << " µs\n";

    double throughput = (double)(total_points * max_iterations) / (total_us / 1e6);
    double latency = (double)total_us / (total_points * max_iterations);
    cout << "PHASE 2 THROUGHPUT = " << throughput << " points per second\n";
    cout << "PHASE 2 LATENCY = " << latency << " µs per point\n";

    return 0;
}
