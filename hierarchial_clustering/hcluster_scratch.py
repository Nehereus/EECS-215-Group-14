import numpy as np
import matplotlib.pyplot as plt

# Compute pairwise Euclidean distances
def compute_distances(data):
    n = len(data)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):  # No need to calculate twice for i and j
            distances[i, j] = distances[j, i] = np.linalg.norm(data[i] - data[j])
    return distances

# Find the two closest clusters and merge them
def find_and_merge(distances, clusters):
    n = len(distances)
    min_dist = float('inf')
    closest_pair = None

    # Find the closest pair of clusters
    for i in range(n):
        for j in range(i+1, n):
            if distances[i, j] < min_dist:
                min_dist = distances[i, j]
                closest_pair = (i, j)

    # Merge the clusters
    c1, c2 = closest_pair
    clusters[c1] = clusters[c1] + clusters[c2]  # Merge into the first cluster
    del clusters[c2]  # Remove the second cluster

    # Update distances after merging
    new_distances = np.delete(np.delete(distances, c2, axis=0), c2, axis=1)
    for i in range(len(new_distances)):
        if i != c1:
            # Calculate new distance for the merged cluster
            new_distances[c1, i] = new_distances[i, c1] = min(distances[c1, i], distances[c2, i])
    return new_distances, clusters, min_dist

# Perform hierarchical clustering
def hierarchical_clustering(data):
    distances = compute_distances(data)
    clusters = {i: [i] for i in range(len(data))}  # Each point starts as its own cluster
    history = []  # To record the merges and distances

    while len(clusters) > 1:
        distances, clusters, dist = find_and_merge(distances, clusters)
        history.append((list(clusters.keys()), dist))

    return history

# Plot the dendrogram
def plot_dendrogram(history):
    from scipy.cluster.hierarchy import dendrogram

    # Convert history to linkage format
    linkage_matrix = []
    cluster_id = len(history) + 1
    for i, (clusters, dist) in enumerate(history):
        c1, c2 = clusters[-2:]  # The last two clusters merged
        linkage_matrix.append([c1, c2, dist, len(clusters)])
        cluster_id += 1

    linkage_matrix = np.array(linkage_matrix, dtype=float)
    dendrogram(linkage_matrix)
    plt.title("Dendrogram")
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.show()

# Visualize clusters in 2D
def plot_clusters(data, clusters):
    plt.figure(figsize=(8, 6))
    for cluster_label, indices in clusters.items():
        cluster_points = data[indices]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_label}")
    plt.title("Clusters in 2D")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()

# Generate random 2D points
np.random.seed(42)
data = np.random.rand(20, 2)

# Perform clustering and visualize
history = hierarchical_clustering(data)
plot_dendrogram(history)

# Assign cluster labels and plot
clusters = {i: [i] for i in range(len(data))}
for cluster_label, _ in enumerate(history[-1][0]):
    clusters[cluster_label] = history[-1][0]
plot_clusters(data, clusters)

