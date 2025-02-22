// Implementation of the KMeans Algorithm
// reference: https://github.com/marcoscastro/kmeans

// SUMMARY
// This optimized K-Means implementation improves the baseline by reducing redundant computations, using loop unrolling, avoiding unnecessary function calls, and leveraging memory optimizations (e.g., `reserve()`, `shrink_to_fit()`, `unordered_set`). 
// Additional improvements include replacing `pow(x, 2.0)` with direct multiplication, avoiding unnecessary `sqrt()` calculations, and using `emplace_back()` for efficiency in vector operations. 
// Samir's code


#include <iostream>		 // For input and output operations (cin, cout)
#include <vector>		 // For using dynamic arrays (vectors)
#include <math.h>		 // For mathematical functions (like pow, sqrt)
#include <stdlib.h>		 // For random number generation (rand, srand)
#include <time.h>		 // For setting the seed of rand()
#include <algorithm>	 // For utility functions like find()
#include <chrono>		 // For measuring execution time
#include <unordered_set> // For faster duplicate checking

using namespace std; // Allows using standard C++ functions without the "std::" prefix

// ============================================================================
//                              Point Class
// ============================================================================
// This class represents a **single data point** in the dataset.
// Each point contains:
//   - A unique **ID** (`id_point`) for identification
//   - A **vector of values** (`values`), which represents feature values
//   - The **total number of feature values** (`total_values`)
//   - A **cluster assignment** (`id_cluster`) to track which cluster it belongs to
//   - An **optional name** (`name`) for identification (e.g., labeling points)

class Point
{
private:
	int id_point;		   // Unique identifier for the point
	int id_cluster;		   // ID of the cluster this point is assigned to
	vector<double> values; // Stores the feature values of the point
	int total_values;	   // Number of features (dimensions) for this point
	string name;		   // Optional name of the point (default: empty)

public:
	// ============================================================================
	// Constructor: Initializes a point with an ID, feature values, and an optional name.
	// ============================================================================
	Point(int id_point, vector<double> &values, string name = "")
	{
		this->id_point = id_point;	  // Assigns the point ID
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

class Cluster
{
private:
	int id_cluster;
	vector<double> central_values;
	vector<Point> points;

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

		points.push_back(point);
	}

	void addPoint(Point point)
	{
		if (points.capacity() == 0) // SAMIR - ✅ Only reserve once
			points.reserve(50);		// dependent based on dataset I am using
									// total points/K is the amount of points in each cluster and we should reserve that amount
		points.push_back(point);	// Efficiently add point
	}

	bool removePoint(int id_point)
	{
		int total_points = points.size();

		for (int i = 0; i < total_points; i++)
		{
			if (points[i].getID() == id_point)
			{
				points.erase(points.begin() + i);
				return true;
			}
		}
		return false;
	}

	// SAMIR - ✅ Inline small getter functions,
	// DON'T INLINE:
	// large functions → Can increase binary size & reduce cache efficiency.
	// recursive functions → Inline won't work well with recursion.
	// virtual functions → Virtual functions use dynamic binding, so inlining doesn't apply.
	// functions that are rarely called → Inlining them gives no performance benefit.

	inline double getCentralValue(int index) const { return central_values[index]; }
	inline void setCentralValue(int index, double value) { central_values[index] = value; }
	inline Point getPoint(int index) const { return points[index]; }
	inline int getTotalPoints() const { return points.size(); }
	inline int getID() const { return id_cluster; }
	inline void shrinkPoints() { points.shrink_to_fit(); }
};

// ==========================================================================
//                              KMeans Class
// ==========================================================================
// This class implements the K-Means clustering algorithm. It follows these steps:
// 1. **Initialize K clusters** by randomly selecting K data points as initial centroids.
// 2. **Assign each data point** to the nearest centroid.
// 3. **Recalculate the centroids** based on the mean of assigned points.
// 4. **Repeat** until either:
//    - The cluster assignments do not change (convergence).
//    - The maximum number of iterations is reached.
//
// The class includes methods for finding the nearest cluster center, running
// the clustering process, and displaying the results.

class KMeans
{
private:
	int K;					  // Number of clusters
	int total_values;		  // Number of features per data point
	int total_points;		  // Total number of data points
	int max_iterations;		  // Maximum iterations allowed
	vector<Cluster> clusters; // Stores the clusters

	// ======================================================================
	// getIDNearestCenter
	// ======================================================================
	// Finds the **nearest cluster** to a given data point using **Euclidean distance**.
	// Returns the index of the closest cluster.
	//
	// The Euclidean distance formula:
	//    distance = sqrt((x1 - x2)^2 + (y1 - y2)^2 + ... + (n1 - n2)^2)
	//
	// It calculates the distance between the given point and each cluster centroid,
	// then selects the closest one.
	// ======================================================================
	int getIDNearestCenter(Point point)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;

		// SAMIR - Loop unrolling
		// Compute distance to the **first cluster** (used as reference)
		int j = 0;
		for (; j + 3 < total_values; j += 4) // Process 4 values per iteration
		{
			double diff1 = clusters[0].getCentralValue(j) - point.getValue(j);
			double diff2 = clusters[0].getCentralValue(j + 1) - point.getValue(j + 1);
			double diff3 = clusters[0].getCentralValue(j + 2) - point.getValue(j + 2);
			double diff4 = clusters[0].getCentralValue(j + 3) - point.getValue(j + 3);
			// SAMIR - Replace pow(x, 2.0) with Direct Multiplication
			sum += (diff1 * diff1) + (diff2 * diff2) + (diff3 * diff3) + (diff4 * diff4);
		}

		// Handle remaining values (if total_values % 4 != 0)
		for (; j < total_values; j++)
		{
			double diff = clusters[0].getCentralValue(j) - point.getValue(j);
			sum += diff * diff;
		}

		min_dist = sum; // SAMIR - Set the first cluster's squared distance as the minimum

		// SAMIR - Compare the Euclidean distance with other clusters
		for (int i = 1; i < K; i++)
		{
			sum = 0.0;

			// Compute the squared Euclidean distance for each cluster
			for (int j = 0; j < total_values; j++)
			{
				double diff = clusters[i].getCentralValue(j) - point.getValue(j);
				sum += diff * diff; // ✅ Use squared difference
			}

			// SAMIR - ✅ No need for sqrt(), just compare squared distances
			if (sum < min_dist)
			{
				min_dist = sum;
				id_cluster_center = i;
			}
		}

		return id_cluster_center; // Return the nearest cluster index
	}

public:
	// ======================================================================
	// Constructor: Initializes the KMeans object with the required parameters.
	// ======================================================================
	KMeans(int K, int total_points, int total_values, int max_iterations)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
	}

	// ======================================================================
	// run
	// ======================================================================
	// Executes the **K-Means clustering algorithm**.
	// It initializes the clusters, assigns points, recalculates centroids,
	// and stops when convergence is reached or the max iterations are exceeded.
	// ======================================================================
	void run(vector<Point> &points)
	{
		auto begin = chrono::high_resolution_clock::now(); // Start total time measurement

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
				points[index_point].setCluster(chosen_indexes.size() - 1);			   // Assign cluster
				clusters.emplace_back(chosen_indexes.size() - 1, points[index_point]); // SAMIR - ✅ Efficiently construct cluster
			}
		}

		auto end_phase1 = chrono::high_resolution_clock::now(); // End of initialization phase

		int iter = 1;
		long long total_iteration_time = 0; // Store cumulative iteration time

		// Step 2: **Iterate until convergence or max_iterations reached**
		while (true)
		{
			auto iteration_start = chrono::high_resolution_clock::now(); // Start iteration time

			bool done = true; // Track whether clusters have stabilized

			// Step 2a: **Assign each point to the nearest cluster**
			for (int i = 0; i < total_points; i++)
			{
				int id_old_cluster = points[i].getCluster();
				int id_nearest_center = getIDNearestCenter(points[i]);

				if (id_old_cluster != id_nearest_center)
				{
					if (id_old_cluster != -1)
						clusters[id_old_cluster].removePoint(points[i].getID());

					points[i].setCluster(id_nearest_center);
					clusters[id_nearest_center].addPoint(points[i]);
					done = false;
				}
			}

			// Step 2b: **Recalculate the centroids based on new assignments**
			for (int i = 0; i < K; i++)
			{ // SAMIR - Loop unrolling
				for (int j = 0; j < total_values; j++)
				{
					int total_points_cluster = clusters[i].getTotalPoints();
					double sum = 0.0;

					if (total_points_cluster > 0)
					{
						int p = 0;
						for (; p + 3 < total_points_cluster; p += 4) // Unroll loop for every 4 points
						{
							sum += clusters[i].getPoint(p).getValue(j) +
								   clusters[i].getPoint(p + 1).getValue(j) +
								   clusters[i].getPoint(p + 2).getValue(j) +
								   clusters[i].getPoint(p + 3).getValue(j);
						}

						// Handle remaining points
						for (; p < total_points_cluster; p++)
						{
							sum += clusters[i].getPoint(p).getValue(j);
						}

						clusters[i].setCentralValue(j, sum / total_points_cluster);
					}
				}
			}

			auto iteration_end = chrono::high_resolution_clock::now();													  // End iteration time
			total_iteration_time += chrono::duration_cast<chrono::microseconds>(iteration_end - iteration_start).count(); // Accumulate iteration time

			// Step 2c: **Check stopping conditions**
			if (done == true || iter >= max_iterations)
			{
				cout << "Break in iteration " << iter << "\n\n";
				break;
			}

			iter++; // Increment iteration count
		}

		auto end = chrono::high_resolution_clock::now(); // End total execution time

		for (int i = 0; i < K; i++)
		{
			clusters[i].shrinkPoints(); // SAMIR - ✅ Reduce memory after clustering is done
		}

		// Step 3: **Display the final clusters and execution time**
		for (int i = 0; i < K; i++)
		{
			int total_points_cluster = clusters[i].getTotalPoints();

			cout << "Cluster " << clusters[i].getID() + 1 << endl;
			for (int j = 0; j < total_points_cluster; j++)
			{
				cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
				for (int p = 0; p < total_values; p++)
					cout << clusters[i].getPoint(j).getValue(p) << " ";

				string point_name = clusters[i].getPoint(j).getName();
				if (point_name != "")
					cout << "- " << point_name;

				cout << endl;
			}

			cout << "Cluster values: ";
			for (int j = 0; j < total_values; j++)
				cout << clusters[i].getCentralValue(j) << " ";

			cout << "\n\n";
		}

		// Display execution time breakdown
		cout << "TOTAL EXECUTION TIME = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << " µs\n";
		cout << "TIME PHASE 1 = " << chrono::duration_cast<chrono::microseconds>(end_phase1 - begin).count() << " µs\n";
		cout << "TIME PHASE 2 = " << chrono::duration_cast<chrono::microseconds>(end - end_phase1).count() << " µs\n";

		// Calculate and display the **average time per iteration**
		if (iter > 1) // Only compute if we have at least 1 iteration
		{
			double avg_time_per_iteration = (double)chrono::duration_cast<chrono::microseconds>(end - end_phase1).count() / iter;
			cout << "FAST-SERIAL, AVERAGE TIME PER ITERATION = " << avg_time_per_iteration << " µs\n";
		}
	}
};

// ==========================================================================
//                              Main Function
// ==========================================================================
// The `main` function is responsible for:
// 1. **Reading user input** to get the number of points, features, clusters, etc.
// 2. **Initializing data points** by storing feature values.
// 3. **Running the K-Means clustering algorithm** with the given parameters.
//
// The program expects input from standard input (cin) in the following format:
//    total_points total_values K max_iterations has_name
//    value1 value2 ... valueN [optional: name]
//    value1 value2 ... valueN [optional: name]
//    ...
//
// Example input format (if has_name = 1):
//    5 2 2 100 1
//    1.0 2.0 PointA
//    3.0 4.0 PointB
//    5.0 6.0 PointC
//    7.0 8.0 PointD
//    9.0 10.0 PointE
//
// Example input format (if has_name = 0):
//    5 2 2 100 0
//    1.0 2.0
//    3.0 4.0
//    5.0 6.0
//    7.0 8.0
//    9.0 10.0
// ==========================================================================
int main(int argc, char *argv[])
{
	// Seed the random number generator (for selecting initial centroids randomly)
	srand(69);

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
	string point_name;			  // To store the optional name of the point

	// ==========================================================================
	// Step 2: Read Points from Input
	// ==========================================================================
	for (int i = 0; i < total_points; i++)
	{
		vector<double> values;		  // Store feature values of the current point
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
			Point p(i, values);				// Create a Point without a name
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
