import re
import csv
import os

# Define input file (results file)
input_filename = "results.txt"

# Regular expressions
implementation_pattern = re.compile(r"===== Running (.+) on .+ =====")
point_pattern = re.compile(r"Point (\d+): (.+)")  # Captures all feature values after the point ID
cluster_pattern = re.compile(r"Cluster (\d+)")
name_pattern = re.compile(r"-\s(.+)")  # Detects name after "-"

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
            implementations[current_impl] = {"data": [], "has_names": False}
            continue

        # Detect cluster start
        match_cluster = cluster_pattern.search(line)
        if match_cluster and current_impl:
            current_cluster = match_cluster.group(1)  # Store numeric cluster ID
            continue

        # Extract points with all available features
        match_point = point_pattern.search(line)
        if match_point and current_impl and current_cluster:
            point_id, feature_data = match_point.groups()

            # Check if there's a name after the "-"
            name_match = name_pattern.search(feature_data)
            point_name = None  # Default: no name

            if name_match:
                point_name = name_match.group(1)  # Extract name
                feature_data = feature_data.split(" - ")[0]  # Remove name from feature list
                implementations[current_impl]["has_names"] = True  # At least one name exists

            # Convert feature_data into a list of numbers
            feature_list = feature_data.split()

            # Store data: Point ID, Features, (optional Name), Cluster
            row_data = [point_id] + feature_list
            if implementations[current_impl]["has_names"]:  # Add name column only if necessary
                row_data.append(point_name if point_name else "")  # Ensure consistent column structure
            row_data.append(current_cluster)  # Append numeric cluster ID
            implementations[current_impl]["data"].append(row_data)

# Create output directory
output_dir = "cluster_results"
os.makedirs(output_dir, exist_ok=True)

# Generate CSV files for each implementation
for impl, impl_data in implementations.items():
    data = impl_data["data"]
    has_names = impl_data["has_names"]

    output_filename = os.path.join(output_dir, f"{impl}_clusters.csv")

    if not data:
        print(f"⚠️ No data found for {impl}. Skipping CSV generation.")
        continue

    # Determine the number of feature columns dynamically
    num_features = len(data[0]) - (3 if has_names else 2)  # Subtract 3 (Point ID, Name, Cluster) if names exist, else 2
    feature_headers = [f"Feature {i+1}" for i in range(num_features)]

    # Construct the header dynamically
    header = ["Point ID"] + feature_headers
    if has_names:
        header.append("Name")  # Only add "Name" if any row has a name
    header.append("Cluster")  # Ensure "Cluster" is always included

    with open(output_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Write dynamic header
        writer.writerows(data)

    print(f"✅ CSV file generated: {output_filename}")
