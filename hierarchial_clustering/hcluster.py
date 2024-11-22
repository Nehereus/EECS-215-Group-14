import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Generate random 2D points
np.random.seed(42)  # Set seed for reproducibility
data = np.random.rand(20, 2)  # 20 points in 2D

# Perform hierarchical clustering
Z = linkage(data, method='ward')  # 'ward' minimizes variance within clusters

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z, labels=np.arange(1, 21), leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

# Assign cluster labels
threshold = 1.5  # Set threshold for maximum cluster distance
clusters = fcluster(Z, threshold, criterion='distance')  # Cluster assignments

# Plot the clustered data
plt.figure(figsize=(8, 6))
for cluster_label in np.unique(clusters):
    cluster_points = data[clusters == cluster_label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}')
plt.title('Clusters in 2D Space')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.show()

