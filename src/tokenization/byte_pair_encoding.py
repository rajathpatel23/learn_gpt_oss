import matplotlib.pyplot as plt
import networkx as nx

# Define merges in order (from example)
merges = [
    ("l", "o", "lo"),
    ("lo", "w", "low"),
    ("e", "r", "er"),
    ("w", "e", "we"),
]

# Create directed graph
G = nx.DiGraph()

# Add merges as edges
for left, right, merged in merges:
    G.add_node(left)
    G.add_node(right)
    G.add_node(merged)
    G.add_edge(left, merged)
    G.add_edge(right, merged)

# Draw the graph
plt.figure(figsize=(8,6))
pos = nx.spring_layout(G, seed=42)  # layout for consistency
nx.draw(G, pos, with_labels=True, node_size=2500, node_color="lightblue", font_size=12, arrowsize=15)
plt.title("Byte Pair Encoding Merge Steps", fontsize=14)
plt.show()
