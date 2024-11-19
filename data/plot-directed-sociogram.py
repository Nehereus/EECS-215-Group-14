import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re

# Set up argument parsing to accept a JSON file input
parser = argparse.ArgumentParser(description='Draw directed graphs with curved, color-coded edges from a JSON file.')
parser.add_argument('json_file', type=str, help='Path to the JSON file containing directed graph data')
args = parser.parse_args()

# Load the JSON data from the specified file
with open(args.json_file, 'r') as f:
    data = json.load(f)

# Sort graphs by the group number extracted from the "id" field
def extract_group_number(graph_id):
    match = re.search(r'group-(\d+)', graph_id)
    return int(match.group(1)) if match else float('inf')  # If no group number, place it at the end

sorted_graphs = sorted(data["graphs"], key=lambda x: extract_group_number(x["id"]))

# Number of graphs to plot
num_graphs = len(sorted_graphs)

# Determine the grid layout for subplots (4 columns, more rows as needed)
cols = 4
rows = (num_graphs + cols - 1) // cols  # This rounds up for any remainder

# Find the maximum edge weight across all graphs to normalize edge thickness
max_weight = max(
    edge["metadata"]["weight"]
    for graph_data in sorted_graphs
    for edge in graph_data["edges"]
)

# Define positions for four corners for consistent node placement
corner_positions = {
    'A': (-1, 1),    # Top-left
    'B': (1, 1),     # Top-right
    'C': (-1, -1),   # Bottom-left
    'D': (1, -1)     # Bottom-right
}

# Create a new figure with subplots
fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
axes = axes.flatten()  # Flatten axes array for easy iteration

# Loop through each sorted graph and plot in a subplot
for i, graph_data in enumerate(sorted_graphs):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with labels
    for node in graph_data["nodes"]:
        G.add_node(node)

    # Add directed edges with weights
    for edge in graph_data["edges"]:
        source = edge["source"]
        target = edge["target"]
        weight = edge["metadata"]["weight"]
        G.add_edge(source, target, weight=weight)

    # Use predefined corner positions for node layout
    pos = {node: corner_positions.get(node, (0, 0)) for node in G.nodes()}

    # Generate a unique color for each node's outgoing edges
    node_colors = {node: color for node, color in zip(G.nodes, cm.tab20.colors)}
    
    # Draw nodes and labels (node ID only)
    nx.draw_networkx_nodes(G, pos, ax=axes[i], node_color='lightcoral', node_size=1000)
    nx.draw_networkx_labels(G, pos, ax=axes[i], labels={node: node for node in G.nodes()}, font_size=12)

    # Draw directed edges with curved connections, unique colors, and edge labels (weights)
    for edge in G.edges(data=True):
        source, target, data = edge
        color = node_colors[source]  # Color based on the source node
        thickness = (data['weight'] / max_weight) * 25  # Scale thickness based on max weight
        
        # Draw the edge with increased curvature
        nx.draw_networkx_edges(
            G, pos, edgelist=[(source, target)], ax=axes[i], width=thickness, 
            arrowstyle='->', arrowsize=15, edge_color=[color], 
            connectionstyle="arc3,rad=0.2"  # Moderate curvature to avoid overlap
        )
        
        # Format the edge label to two decimal places
        edge_label = f"{data['weight']:.2f}"

        # Position the edge label to the right of the edge
        x1, y1 = pos[source]
        x2, y2 = pos[target]
        offset_x = (x2 - x1) * 0.1
        offset_y = (y2 - y1) * 0.1
        label_pos_x = (x1 + x2) / 2 + offset_x
        label_pos_y = (y1 + y2) / 2 + offset_y

        # Use ax.annotate for better control over label positioning
        axes[i].annotate(
            text=edge_label,
            xy=(label_pos_x, label_pos_y), 
            color=color, 
            fontsize=10, 
            ha='center'
        )

    # Set title and hide axes
    axes[i].set_title(graph_data["label"])
    axes[i].axis('off')

    # Adjust axis limits to fit nodes and edges within each subplot
    axes[i].set_xlim(-1.5, 1.5)
    axes[i].set_ylim(-1.5, 1.5)

# Hide any remaining empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust layout and display all graphs
plt.tight_layout()
plt.show()
