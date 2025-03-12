# hw2_kmeans

The purpose of this program is to take a sequential implementation of the K-Means clustering algorithm and optimize it for improved performance by applying parallelization techniques on the CPU. The project explores a wide range of fine-grained performance improvements including Intel Threading Building Blocks (TBB), with a focus on both scalability and computational efficiency.

## How to run
./run.sh <implementation(s)> <dataset.txt>

Examples are as follows:  
./run.sh s 1.txt  
Runs the serial implementation on the dataset 1.txt  

./run.sh s f 2.txt   
Runs the serial and fast-serial implementation on the dataset 2.txt  

./run.sh s f n l 3.txt   
Runs the serial, fast-serial, na-serial, and lightning-serial implementation on the dataset 3.txt  

./run.sh a b p 8.txt  
Runs the a-parallel, b-parallel, and parallel implementation on the dataset 8.txt

[s]="src/serial.cpp serial"  
[f]="src/fast-serial.cpp fast-serial"  
[p]="src/parallel.cpp parallel"  
[n]="src/na-serial.cpp na-serial"  
[l]="src/lightning-serial.cpp lightning-serial"  
[a]="src/a-parallel.cpp a-parallel"  
[b]="src/b-parallel.cpp b-parallel"  
[u]="src/usion-parallel.cpp usion-parallel"  

## Understanding the output
Example output:  

parallel:
  - Time Phase 2: 22355 µs
  - Iterations: 177
  - Average Time per Iteration: 126.299 µs
  - Throughput (Phase 2): 1.07768e+08 points per second
  - Latency (Phase 2): 0.00927922 µs per point
  - Final Cluster Values: 57673.6 947.431 371.248 200.14 1.87332 0.832684 58541.7 270.904 0.720855 0.98521 0.808614 0.733245 0.00644676 0.00116156 0.540491 0.992616 

Time Phase 2 is the total execution time spent in Phase 2, which includes: Cluster assignment (Step 2a), Centroid recomputation (Step 2b). It excludes Phase 1, which is initial centroid selection.
Iterations is the number of iterations it took for the K-Means algorithm to reach convergence (i.e., no points moved to a different cluster or the max_iterations was hit).
Average Time per Iteration is the Phase 2 total time divided by number of iterations.
Throughput (Phase 2) is the rate at which points were processed during Phase 2.
Latency (Phase 2) is the average time it took to process one point per iteration.
Final Cluster Values is the final centroid positions for your clusters.

## Explanation of source code
a-parallel.cpp -> This version of the K-Means clustering algorithm introduces parallelization using Intel TBB to speed up execution and improve scalability. (Step 2a)  

b-parallel.cpp -> This version of the K-Means clustering algorithm further enhances **parallelization using Intel TBB by optimizing the centroid recalculation step (Step 2b)

fast-serial.cpp -> This optimized K-Means implementation improves the baseline by reducing redundant computations, using loop unrolling, avoiding unnecessary function calls, and leveraging memory optimizations  

lightning-serial.cpp -> This optimized K-Means implementation enhances both performance and memory efficiency by eliminating per-cluster point storage, maintaining only centroid values, and recalculating centroids using aggregate sums

na-serial.cpp -> This version of K-Means optimizes memory usage by removing per-cluster point storage, keeping only centroid values, and recalculating centroids using aggregate sums.

parallel.cpp -> This version of the K-Means clustering algorithm **fully parallelizes both cluster assignment and centroid recomputation using Intel TBB.  Combines Steps 2a and 2b

serial.cpp -> This is the baseline implementation of the K-Means clustering algorithm, measuring execution time and average time per iteration. It initializes clusters randomly, assigns points based on Euclidean distance, recalculates centroids iteratively, and stops upon convergence or reaching the maximum iterations. This is the Professor's code.

## Datasets chosen
Metadata is present on top of each .txt dataset file. The metadata was added after the dataset was downloaded.  

1st number represents total number of points  
2nd number represents number of features/dimensions per point  
3rd number represents number of clusters K to generate  
4th number represents maximum number of iterations allowed for the K-Means algorithm  
5th number represents whether the point has a name (a boolean flag: 0 = no names, 1 = names are present for each point)  

See [datasets/datasets.md](datasets/datasets.md) for dataset information.

## Charts
https://docs.google.com/spreadsheets/d/1w9QckUbBnYQO1gXstozWQDwRfLRCrpcjpW0PHK8apVI/edit?usp=sharing

There are execution times and charts present at this link. We compare the average time per iteration when we vary the amount of points, amount of dimensions, and amount of clusters. We also compare the throughput (points per second) and latency (microseconds per point) when we vary the amount of dimensions and the amount of clusters.

## Notes
- This was run on a Sunlab machine at Lehigh University with 16 CPUs (try 'less /proc/cpuinfo'), therefore results produced on another machine might not compare
- The Sunlab computers have a specific configuration that might not be replicable on other machines
- In each .cpp file, you'll see that srand() is initialized with a fixed value (e.g., srand(10)). This is done to ensure reproducibility — it guarantees that the same set of initial centroids is selected every time the program runs. While we can randomize the seed (e.g., using srand(time(NULL))) to get different initial cluster centers in each run, fixing it helps maintain consistent output for benchmarking and debugging purposes.
