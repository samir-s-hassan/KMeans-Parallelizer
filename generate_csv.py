import re
import csv
import os

# Define input file (results file)
input_filename = "results.txt"

# Regular expressions
implementation_pattern = re.compile(r"===== Running (.+) on .+ =====")
point_pattern = re.compile(r"Point (\d+): (.+)")  # Capture all feature values after the point ID
cluster_pattern = re.compile(r"Cluster (\d+)")

# Storage for results
implementations = {}

with open(input_filename, "r") as file:
    current_impl = None
    current_cluster = None

    for line in file:
        # Detect new implementation
        match_impl = implementation_pattern.search(line)
        if match_impl:
            current_impl = match_impl.group(1)
            implementations[current_impl] = []
            continue

        # Detect cluster start
        match_cluster = cluster_pattern.search(line)
        if match_cluster and current_impl:
            current_cluster = match_cluster.group(1)
            continue

        # Extract points with all available features
        match_point = point_pattern.search(line)
        if match_point and current_impl and current_cluster:
            point_id, features = match_point.groups()
            feature_list = features.split()  # Split all numerical values
            implementations[current_impl].append([point_id] + feature_list + [current_cluster])

# Create output directory
output_dir = "cluster_results"
os.makedirs(output_dir, exist_ok=True)

# Generate CSV files for each implementation
for impl, data in implementations.items():
    output_filename = os.path.join(output_dir, f"{impl}_clusters.csv")

    if not data:
        print(f"⚠️ No data found for {impl}. Skipping CSV generation.")
        continue

    # Determine the number of feature columns dynamically
    num_features = len(data[0]) - 2  # Subtract 2 (Point ID & Cluster columns)
    feature_headers = [f"Feature {i+1}" for i in range(num_features)]

    with open(output_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Point ID"] + feature_headers + ["Cluster"])  # Dynamic header
        writer.writerows(data)

    print(f"CSV file generated: {output_filename}")