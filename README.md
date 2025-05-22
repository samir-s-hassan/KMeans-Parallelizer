# KMeans-Parallelizer

The purpose of this program is to take a sequential implementation of the K-Means clustering algorithm and optimize it for improved performance by applying parallelization techniques on the CPU. The project explores a wide range of fine-grained performance improvements including Intel Threading Building Blocks (TBB), with a focus on both scalability and computational efficiency.  

There is also a comparison done between GPU code (km_cuda implementation) and my fastest parallel CPU codes (a-parallel and parallel). We test these versions on large datasets/varied datasets to see the effects of GPU cores (many lightweight threads) on the computations involved in the KMeans algorithm.

## How to run
./km_run.sh <implementation(s)> <dataset.txt>  
- use only for kmcuda implementations vs parallel implementations  
  
./run.sh <implementation(s)> <dataset.txt>   
- use for all other purposes  

## Example commands of how to run
./run.sh s 1.txt  
- Runs the serial implementation on the dataset 1.txt  

./run.sh s f 2.txt   
- Runs the serial and fast-serial implementations on the dataset 2.txt  

./run.sh s f n l 3.txt   
- Runs the serial, fast-serial, na-serial, and lightning-serial implementations on the dataset 3.txt  

./run.sh a b p 8.txt  
- Runs the a-parallel, b-parallel, and parallel implementations on the dataset 8.txt  

./km_run.sh k r p a b datasets/9.txt  
- Runs the kmcuda, archived kmcuda, parallel, a-parallel, and b-parallel implementations on the dataset 9.txt (NOTE: specify datasets/ because this script doesn't have datasets/ as the root directory for datasets)  

./km_run.sh k p a datasets/9.txt  
- Runs the kmcuda, parallel, and a-parallel implementations on the dataset 9.txt (NOTE: specify datasets/ because this script doesn't have datasets/ as the root directory for datasets)  

./km_run.sh k datasets/9.txt  
- Runs the kmcuda implementation on the dataset 9.txt (NOTE: specify datasets/ because this script doesn't have datasets/ as the root directory for datasets)  

s = src/serial.cpp  
f = src/fast-serial.cpp  
p = src/parallel.cpp  
n = src/na-serial.cpp  
l = src/lightning-serial.cpp  
a = src/a-parallel.cpp  
b = src/b-parallel.cpp  
u = src/usion-parallel.cpp  
k = src/kmcuda_wrapper.cpp  
r = src/[ARCHIVED]kmc.cpp

## Understanding the output
Example output:  

parallel:
  - Time Phase 2: 22355 µs
  - Iterations: 177
  - Average Time per Iteration: 126.299 µs
  - Throughput (Phase 2): 1.07768e+08 points per second
  - Latency (Phase 2): 0.00927922 µs per point
  - Final Cluster Values: 57673.6 947.431 371.248 200.14 1.87332 0.832684 58541.7 270.904 0.720855 0.98521 0.808614 0.733245 0.00644676 0.00116156 0.540491 0.992616 

Time Phase 2 is the total execution time spent in Phase 2, which includes: Cluster assignment (Step 2a), Centroid recomputation (Step 2b); it excludes Phase 1, which is initial centroid selection.  
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

kmeans_wrapper.cpp -> This version integrates the KM-CUDA GPU-accelerated KMeans algorithm into the project.
It reads and formats input data, invokes kmeans_cuda() to perform clustering entirely on the GPU, and intercepts the library's output using POSIX pipe() to measure the number of iterations.

[ARCHIVED]kmc.cpp -> This version uses the KM-CUDA library to run KMeans clustering entirely on the GPU with minimal overhead. It reads input data, executes kmeans_cuda() with predefined parameter. Unlike the full wrapper, it does not capture iteration count or split execution into phases.

usion-parallel.cpp -> DOES NOT WORK, ignore this file

## Datasets chosen
Metadata is present on top of each .txt dataset file. The metadata was added after the dataset was downloaded.  

1st number represents total number of points  
2nd number represents number of features/dimensions per point  
3rd number represents number of clusters K to generate  
4th number represents maximum number of iterations allowed for the K-Means algorithm  
5th number represents whether the point has a name (a boolean flag: 0 = no names, 1 = names are present for each point)  

See [datasets/datasets.md](datasets/datasets.md) for dataset information.

## Charts
View the chart:
1. [On Google Sheets](https://docs.google.com/spreadsheets/d/1w9QckUbBnYQO1gXstozWQDwRfLRCrpcjpW0PHK8apVI/edit?usp=sharing)  
2. [Download File](./charts/chart1.xlsx)

There are execution times and charts present at this link (if the link does not work, you can download or view the file by clicking the second option.) We compare the average time per iteration when we vary the amount of points, amount of dimensions, and amount of clusters. We also compare the throughput (points per second) and latency (microseconds per point) when we vary the amount of dimensions and the amount of clusters.

1. [On Google Sheets](https://docs.google.com/spreadsheets/d/1VlR9bbsMi0Q9YT6g9q1Kq0vtMYysejJe5KwZs9Vb620/edit?usp=sharing  )  
2. [Download File](./charts/chart2.xlsx)

This link is to test the km_cuda implementation, which is GPU code, against the a-parallel and parallel implementations, which are CPU code (if the link does not work, you can download or view the file by clicking the second option.) There are execution times and charts present at this link. We compare the execution times and average time per iteration when we vary the amount of points and amount of clusters. We also compare the throughput (points per second) and latency (microseconds per point) when we vary the amount of points and the amount of clusters. 

## Notes
- This was run on a Sunlab machine at Lehigh University with 16 CPUs (try 'less /proc/cpuinfo'), therefore results produced on another machine might not compare
- The Sunlab computers have a specific configuration that might not be replicable on other machines
- In each .cpp file, you'll see that srand() is initialized with a fixed value (e.g., srand(10)). This is done to ensure reproducibility — it guarantees that the same set of initial centroids is selected every time the program runs. While we can randomize the seed (e.g., using srand(time(NULL))) to get different initial cluster centers in each run, fixing it helps maintain consistent output for benchmarking and debugging purposes.
- generate_csv.py parses a results.txt file (which contains printed K-Means clustering output), extracts cluster assignments and point features, and generates a clean CSV file per implementation (e.g., serial, b-parallel, etc.) for further analysis or visualization.
- The km_run script was specifically made to test the km_cuda implementation and therefore might not work for all the implementations. It does work perfect though for km_cuda, a-parallel, b-parallel, and parallel implementations.  
- When using datasets/9.txt to test the km_cuda implementation, increase clusters from 4 to a greater number (e.g. 20) as this will allow kmeans++ to kick in. In general, kmeans++ will only kick in when there aren't too few amount of clusters.
- kmcuda/ and oneapi-tbb-2022.0.0/ were downloaded from the internet. Check instructions/ to understand

## License

    Copyright 2025 Samir Hassan

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
