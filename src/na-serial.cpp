// Implementation of the KMeans Algorithm
// reference: https://github.com/marcoscastro/kmeans

// SUMMARY
// This version of K-Means optimizes memory usage by removing per-cluster point storage, keeping only centroid values, and recalculating centroids using aggregate sums. It also simplifies cluster assignment logic, reduces memory overhead, and improves performance by focusing solely on centroid updates rather than maintaining point lists within clusters.
// Samir's code

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>

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
        this->id_point = id_point;
        total_values = values.size();
        this->values = values;
        this->name = name;
        id_cluster = -1; // Initially unassigned
    }

    int getID() { return id_point; }
    void setCluster(int id_cluster) { this->id_cluster = id_cluster; }
    int getCluster() { return id_cluster; }
    double getValue(int index) { return values[index]; }
    int getTotalValues() { return total_values; }
    string getName() { return name; }
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

        for (int i = 0; i < total_values; i++)
            central_values.push_back(point.getValue(i));
    }

    int getID() { return id_cluster; }
    double getCentralValue(int index) { return central_values[index]; }
    void setCentralValue(int index, double value) { central_values[index] = value; }
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
        double min_dist = numeric_limits<double>::max();
        int id_cluster_center = 0;

        for (int i = 0; i < K; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < total_values; j++)
            {
                sum += pow(clusters[i].getCentralValue(j) - point.getValue(j), 2.0);
            }

            double dist = sqrt(sum);
            if (dist < min_dist)
            {
                min_dist = dist;
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

        vector<int> prohibited_indexes;

        // Step 1: **Select K initial centroids randomly**
        for (int i = 0; i < K; i++)
        {
            while (true)
            {
                int index_point = rand() % total_points;
                if (find(prohibited_indexes.begin(), prohibited_indexes.end(), index_point) == prohibited_indexes.end())
                {
                    prohibited_indexes.push_back(index_point);
                    points[index_point].setCluster(i);
                    clusters.push_back(Cluster(i, points[index_point]));
                    break;
                }
            }
        }

        auto end_phase1 = chrono::high_resolution_clock::now();
        int iter = 1;
        long long total_iteration_time = 0;

        // Step 2: **Iterate until convergence or max_iterations reached**
        while (true)
        {
            auto iteration_start = chrono::high_resolution_clock::now();
            bool done = true;

            // Step 2a: **Assign each point to the nearest cluster**
            for (int i = 0; i < total_points; i++)
            {
                int id_old_cluster = points[i].getCluster();
                int id_nearest_center = getIDNearestCenter(points[i]);

                if (id_old_cluster != id_nearest_center)
                {
                    points[i].setCluster(id_nearest_center);
                    done = false;
                }
            }

            // Step 2b: **Recalculate centroids based on new assignments**
            vector<vector<double>> new_centroids(K, vector<double>(total_values, 0.0));
            vector<int> cluster_sizes(K, 0);

            // Sum all point values for each cluster
            for (int i = 0; i < total_points; i++)
            {
                int cluster_id = points[i].getCluster();
                cluster_sizes[cluster_id]++;
                for (int j = 0; j < total_values; j++)
                    new_centroids[cluster_id][j] += points[i].getValue(j);
            }

            // Compute the new centroid values
            for (int i = 0; i < K; i++)
            {
                if (cluster_sizes[i] > 0)
                {
                    for (int j = 0; j < total_values; j++)
                        clusters[i].setCentralValue(j, new_centroids[i][j] / cluster_sizes[i]);
                }
            }

            auto iteration_end = chrono::high_resolution_clock::now();
            total_iteration_time += chrono::duration_cast<chrono::microseconds>(iteration_end - iteration_start).count();

            // Step 2c: **Check stopping condition**
            if (done || iter >= max_iterations)
            {
                cout << "Break in iteration " << iter << "\n\n";
                break;
            }
            iter++;
        }

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
            cout << "NA-SERIAL, AVERAGE TIME PER ITERATION = " << avg_time_per_iteration << " µs\n";
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
    string point_name; // To store the optional name of the point

    // ==========================================================================
    // Step 2: Read Points from Input
    // ==========================================================================
    for (int i = 0; i < total_points; i++)
    {
        vector<double> values; // Store feature values of the current point

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
            points.push_back(p);            // Store the point in the dataset
        }
        else
        {
            Point p(i, values);  // Create a Point without a name
            points.push_back(p); // Store the point in the dataset
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
