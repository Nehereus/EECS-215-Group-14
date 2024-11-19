import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import re

# Set up argument parsing to accept a JSON file input
parser = argparse.ArgumentParser(description='Draw graphs from a JSON file.')
parser.add_argument('json_file', type=str, help='Path to the JSON file containing graph data')
args = parser.parse_args()

# Load the JSON data from the specified file
with open(args.json_file, 'r') as f:
    data = json.load(f)

# Function to extract group number from the "id" field (e.g., "group-10-shared-attention-graph")
def extract_group_number(graph_id):
    match = re.search(r'group-(\d+)', graph_id)
    return int(match.group(1)) if match else float('inf')  # Default to 'inf' if no match

# Sort the graphs based on the group number extracted from the 'id' field
data["graphs"].sort(key=lambda x: extract_group_number(x["id"]))

# Number of graphs to plot
num_graphs = len(data["graphs"])

# Determine the grid layout for subplots (4 columns, more rows as needed)
cols = 4
rows = (num_graphs + cols - 1) // cols  # This rounds up for any remainder

# Find the maximum edge weight across all graphs to normalize edge thickness
max_weight = max(
    edge["metadata"]["weight"]
    for graph_data in data["graphs"]
    for edge in graph_data["edges"]
)

# Define positions for the four corners
corner_positions = {
    'A': (-1, 1),    # Top-left
    'B': (1, 1),     # Top-right
    'C': (-1, -1),   # Bottom-left
    'D': (1, -1)     # Bottom-right
}

# Create a new figure with subplots
fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
axes = axes.flatten()  # Flatten axes array for easy iteration

# Loop through each graph and plot in a subplot
for i, graph_data in enumerate(data["graphs"]):
    G = nx.Graph()

    # Add nodes with identifiers as labels
    for node, details in graph_data["nodes"].items():
        G.add_node(node)

    # Add edges with weights
    for edge in graph_data["edges"]:
        source = edge["source"]
        target = edge["target"]
        weight = edge["metadata"]["weight"]
        G.add_edge(source, target, weight=weight)

    # Use the predefined corner positions for node layout
    pos = {node: corner_positions.get(node, (0, 0)) for node in G.nodes()}

    # Draw nodes and identifiers as labels
    nx.draw_networkx_nodes(G, pos, ax=axes[i], node_color='skyblue', node_size=1000)
    nx.draw_networkx_labels(G, pos, ax=axes[i], labels={node: node for node in G.nodes()}, font_size=12)

    # Draw edges with thickness proportional to their weight
    edge_widths = [(G[u][v]['weight'] / max_weight) * 25 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=axes[i], width=edge_widths, connectionstyle='arc3,rad=0.2')  # Adds slight curvature

    # Edge labels (weights) - Offset for better readability
    edge_labels = {(edge["source"], edge["target"]): edge["metadata"]["weight"] for edge in graph_data["edges"]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=axes[i], font_size=10, 
                                 label_pos=0.3, rotate=True)

    # Set title and hide axes
    axes[i].set_title(graph_data["label"])
    axes[i].axis('off')

# Hide any remaining empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust layout and display all graphs
plt.tight_layout()
plt.show()
