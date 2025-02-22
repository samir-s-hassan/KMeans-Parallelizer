// Implementation of the KMeans Algorithm
// reference: https://github.com/marcoscastro/kmeans

// SUMMARY
// This implementation of the K-Means clustering algorithm applies fusion optimization by combining the reassignment and sum steps using Intel TBB’s parallelization features. It restructures the loop to first compute an initial sum, then iteratively perform divide, reassign, and sum while any points continue to move between clusters, ensuring efficient centroid updates with minimal synchronization overhead.
// Samir's code CURRENTLY BROKEN IN SUBMISSION, WANT TO CONTINUE WORKING ON IT AS APPROACH CODE REVIEW

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <unordered_set>
// parallel
#include <tbb/parallel_for.h>
#include <atomic>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/concurrent_unordered_set.h>

using namespace std;

// ============================================================================
//                              Point Class
// ============================================================================
// This class represents a **single data point** in the dataset.

class Point
{
private:
    int id_point;          // Unique identifier for the point
    int id_cluster;        // ID of the cluster this point is assigned to
    vector<double> values; // Stores the feature values of the point
    int total_values;      // Number of features (dimensions) for this point
    string name;           // Optional name of the point (default: empty)

public:
    Point(int id_point, vector<double> &values, string name = "")
    {
        this->id_point = id_point;    // Assigns the point ID
        total_values = values.size(); // Stores the total number of features
        // this->values.reserve(total_values); // DOESN'T WORK - ✅ Avoids dynamic resizing

        // SAMIR - Loop unrolling
        int i = 0;
        for (; i + 3 < total_values; i += 4) // Copy 4 values per loop
        {
            this->values.push_back(values[i]);
            this->values.push_back(values[i + 1]);
            this->values.push_back(values[i + 2]);
            this->values.push_back(values[i + 3]);
        }

        // Handle remaining values
        for (; i < total_values; i++)
            this->values.push_back(values[i]);

        this->name = name; // Assigns the name (if provided)
        id_cluster = -1;   // Initially, the point is not assigned to any cluster (-1)
    }

    // ============================================================================
    // Getter Methods: Retrieve information about the point.
    // ============================================================================

    // SAMIR - ✅ Inline small functions to reduce function call overhead
    inline int getID() const { return id_point; }
    inline int getCluster() const { return id_cluster; }
    inline void setCluster(int id_cluster) { this->id_cluster = id_cluster; }
    inline double getValue(int index) const { return values[index]; }
    inline int getTotalValues() const { return total_values; }
    inline string getName() const { return name; }
};

// ============================================================================
//                              Cluster Class
// ============================================================================
// Stores only the **centroid values** of a cluster.

class Cluster
{
private:
    int id_cluster;
    vector<double> central_values; // Centroid coordinates

public:
    Cluster(int id_cluster, Point point)
    {
        this->id_cluster = id_cluster;

        int total_values = point.getTotalValues();
        central_values.reserve(total_values); // SAMIR - ✅ Reserve space for feature values

        int i = 0;
        // SAMIR - Unroll by copying 4 feature values at a time
        for (; i + 3 < total_values; i += 4)
        {
            central_values.push_back(point.getValue(i));
            central_values.push_back(point.getValue(i + 1));
            central_values.push_back(point.getValue(i + 2));
            central_values.push_back(point.getValue(i + 3));
        }

        // Copy remaining feature values
        for (; i < total_values; i++)
        {
            central_values.push_back(point.getValue(i));
        }
    }

    inline double getCentralValue(int index) const { return central_values[index]; }
    inline void setCentralValue(int index, double value) { central_values[index] = value; }
    inline int getID() const { return id_cluster; }
};

// ============================================================================
//                              KMeans Class
// ============================================================================
// Implements the K-Means algorithm.

class KMeans
{
private:
    int K;                    // Number of clusters
    int total_values;         // Number of features per point
    int total_points;         // Total number of points
    int max_iterations;       // Maximum iterations allowed
    vector<Cluster> clusters; // Stores only cluster centroids

    // ======================================================================
    // Finds the **nearest cluster** to a given point using **Euclidean distance**.
    // ======================================================================
    int getIDNearestCenter(Point &point)
    {
        // Attempted to parallelize this function but even on big datasets such as 8.txt we get no speedup
        double min_dist_sq = numeric_limits<double>::max(); // Store squared distance
        int id_cluster_center = 0;

        for (int i = 0; i < K; i++)
        {
            double sum = 0.0;
            int j = 0;

            // SAMIR - Process 4 values at a time (Loop Unrolling by 4)
            for (; j + 3 < total_values; j += 4)
            {
                double diff0 = clusters[i].getCentralValue(j) - point.getValue(j);
                double diff1 = clusters[i].getCentralValue(j + 1) - point.getValue(j + 1);
                double diff2 = clusters[i].getCentralValue(j + 2) - point.getValue(j + 2);
                double diff3 = clusters[i].getCentralValue(j + 3) - point.getValue(j + 3);

                // SAMIR - Replace pow(x, 2.0) with Direct Multiplication
                sum += (diff1 * diff1) + (diff2 * diff2) + (diff3 * diff3) + (diff0 * diff0);
            }

            // Process remaining elements (if any)
            for (; j < total_values; j++)
            {
                double diff = clusters[i].getCentralValue(j) - point.getValue(j);
                sum += diff * diff;
            }

            // SAMIR - No sqrt() needed - compare squared distances
            if (sum < min_dist_sq)
            {
                min_dist_sq = sum;
                id_cluster_center = i;
            }
        }
        return id_cluster_center;
    }

public:
    KMeans(int K, int total_points, int total_values, int max_iterations)
    {
        this->K = K;
        this->total_points = total_points;
        this->total_values = total_values;
        this->max_iterations = max_iterations;
    }

    void run(vector<Point> &points)
    {
        auto begin = chrono::high_resolution_clock::now();

        if (K > total_points)
            return;

        unordered_set<int> chosen_indexes; // SAMIR - ✅ Use unordered_set for O(1) lookups

        clusters.reserve(K); // SAMIR - ✅ Reserve memory for K clusters to avoid dynamic resizing

        // Step 1: **Select K unique initial centroids randomly**
        while (chosen_indexes.size() < K)
        {
            int index_point = rand() % total_points;

            if (chosen_indexes.insert(index_point).second) // SAMIR - ✅ O(1) lookup and insert
            {
                points[index_point].setCluster(chosen_indexes.size() - 1);             // Assign cluster
                clusters.emplace_back(chosen_indexes.size() - 1, points[index_point]); // SAMIR - ✅ Efficiently construct cluster
            }
        }

        //^^^ Don't want to parallelize this because Time Phase 1 is very small regardless of dataset and it can mess with rand(). Gets too confusing
        auto end_phase1 = chrono::high_resolution_clock::now();
        long long total_iteration_time = 0;

        // Compute initial sums based on random assignments
        vector<vector<double>> new_centroids(K, vector<double>(total_values, 0.0));
        vector<int> cluster_sizes(K, 0);

        // Precompute the initial cluster sums
        for (size_t i = 0; i < points.size(); ++i)
        {
            int cluster_id = points[i].getCluster();
            cluster_sizes[cluster_id]++;

            for (int j = 0; j < total_values; j++)
            {
                new_centroids[cluster_id][j] += points[i].getValue(j);
            }
        }

        int iter = 1;

        tbb::enumerable_thread_specific<vector<vector<double>>> local_sums;
        tbb::enumerable_thread_specific<vector<int>> local_counts;
        while (true)
        {
            auto iteration_start = chrono::high_resolution_clock::now();
            // === Divide Step: Compute New Centroids ===
            tbb::parallel_for(0, K, [&](int i)
                              {
                if (cluster_sizes[i] > 0) {
                    double inv_size = 1.0 / cluster_sizes[i]; // Precompute division
                    for (int j = 0; j < total_values; j++) {
                        clusters[i].setCentralValue(j, new_centroids[i][j] * inv_size);
                    }
                } });

            std::atomic<bool> done(true);
            // === Fused Reassign + Sum Step ===
            tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()), [&](const tbb::blocked_range<size_t> &r)
                              {
                auto &local_centroids = local_sums.local();
                auto &local_cluster_sizes = local_counts.local();

                for (size_t i = r.begin(); i < r.end(); ++i) {
                    int old_cluster = points[i].getCluster();
                    int new_cluster = getIDNearestCenter(points[i]);

                    if (old_cluster != new_cluster) {
                        points[i].setCluster(new_cluster);
                        done.store(false, std::memory_order_relaxed);
                    }

                    local_cluster_sizes[new_cluster]++;
                    for (int j = 0; j < total_values; j++) {
                        local_centroids[new_cluster][j] += points[i].getValue(j);
                    }
                } });

            // Merge thread-local results
            tbb::parallel_for(0, K, [&](int i)
                              {
                for (const auto &local_centroids : local_sums) {
                    for (int j = 0; j < total_values; j++) {
                        new_centroids[i][j] += local_centroids[i][j];
                    }
                }
                for (const auto &local_cluster_sizes : local_counts) {
                    cluster_sizes[i] += local_cluster_sizes[i];
                } });

            // === Reset for Next Iteration ===
            fill(cluster_sizes.begin(), cluster_sizes.end(), 0);
            for (auto &row : new_centroids)
            {
                fill(row.begin(), row.end(), 0.0);
            }

            auto iteration_end = chrono::high_resolution_clock::now();
            total_iteration_time += chrono::duration_cast<chrono::microseconds>(iteration_end - iteration_start).count();

            // === Check Stopping Condition ===
            if (done.load() || iter >= max_iterations)
            {
                cout << "Break in iteration " << iter << "\n\n";
                break;
            }

            iter++; // Increment iteration count
        }
        // === Final Computation of Centroids (Ensure Consistency) ===
        tbb::parallel_for(0, K, [&](int i)
                          {
            if (cluster_sizes[i] > 0) {
                double inv_size = 1.0 / cluster_sizes[i];
                for (int j = 0; j < total_values; j++) {
                    clusters[i].setCentralValue(j, new_centroids[i][j] * inv_size);
                }
            } });
        auto end = chrono::high_resolution_clock::now();

        // Step 3: **Display results**
        for (int i = 0; i < K; i++)
        {
            cout << "Cluster " << clusters[i].getID() + 1 << endl;
            for (int j = 0; j < total_points; j++)
            {
                if (points[j].getCluster() == i)
                {
                    cout << "Point " << points[j].getID() + 1 << ": ";
                    for (int p = 0; p < total_values; p++)
                        cout << points[j].getValue(p) << " ";
                    string point_name = points[j].getName();
                    if (point_name != "")
                        cout << "- " << point_name;

                    cout << endl;
                }
            }
            cout << "Cluster values: ";
            for (int j = 0; j < total_values; j++)
                cout << clusters[i].getCentralValue(j) << " ";

            cout << "\n\n";
        }

        cout << "TOTAL EXECUTION TIME = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << " µs\n";
        cout << "TIME PHASE 1 = " << chrono::duration_cast<chrono::microseconds>(end_phase1 - begin).count() << " µs\n";
        cout << "TIME PHASE 2 = " << chrono::duration_cast<chrono::microseconds>(end - end_phase1).count() << " µs\n";

        // Calculate and display the **average time per iteration**
        if (iter > 1) // Only compute if we have at least 1 iteration
        {
            double avg_time_per_iteration = (double)chrono::duration_cast<chrono::microseconds>(end - end_phase1).count() / iter;
            cout << "USION-PARALLEL, AVERAGE TIME PER ITERATION = " << avg_time_per_iteration << " µs\n";
        }
    }
};

int main(int argc, char *argv[])
{
    // Seed the random number generator (for selecting initial centroids randomly)
    // srand(time(NULL));
	srand(10);

    int total_points, total_values, K, max_iterations, has_name;

    // ==========================================================================
    // Step 1: Read Input Values
    // ==========================================================================
    // Read the total number of data points, the number of features per point,
    // the number of clusters (K), the maximum number of iterations, and whether
    // each point has a name.
    cin >> total_points >> total_values >> K >> max_iterations >> has_name;

    // Declare a vector to store all points in the dataset
    vector<Point> points;
    points.reserve(total_points); // SAMIR - Preallocate memory for all points
    string point_name;            // To store the optional name of the point

    // ==========================================================================
    // Step 2: Read Points from Input
    // ==========================================================================
    for (int i = 0; i < total_points; i++)
    {
        vector<double> values;        // Store feature values of the current point
        values.reserve(total_values); // SAMIR - ✅ Preallocate memory for feature values

        // Read the feature values for the current point
        for (int j = 0; j < total_values; j++)
        {
            double value;
            cin >> value;
            values.push_back(value);
        }

        // If the points have names, read and store the name
        if (has_name)
        {
            cin >> point_name;
            Point p(i, values, point_name); // Create a Point with a name
            points.emplace_back(i, values); // SAMIR - ✅ Use `emplace_back()` instead of `push_back()`
        }
        else
        {
            Point p(i, values);             // Create a Point without a name
            points.emplace_back(i, values); // SAMIR - ✅ Use `emplace_back()` instead of `push_back()`
        }
    }

    // ==========================================================================
    // Step 3: Initialize K-Means Algorithm and Run Clustering
    // ==========================================================================
    // Create an instance of KMeans with the input parameters
    KMeans kmeans(K, total_points, total_values, max_iterations);

    // Run the K-Means algorithm on the dataset
    kmeans.run(points);

    // ==========================================================================
    // Step 4: Exit Program
    // ==========================================================================
    return 0; // Return 0 to indicate successful execution
}
