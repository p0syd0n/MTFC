"""
analyze.py  –  Visualise training_log.json produced by the Rust GBM.

Run:
    pip install matplotlib networkx
    python analyze.py                          # uses training_log.json by default
    python analyze.py path/to/training_log.json
"""

import json
import sys
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

FOLDER = "graphs/"
# ── Load log ──────────────────────────────────────────────────────────────────

log_path = sys.argv[1] if len(sys.argv) > 1 else "training_log.json"

with open(log_path) as f:
    events = [json.loads(line) for line in f if line.strip()]

init_event  = next(e for e in events if e["event"] == "init")
tree_events = [e for e in events if e["event"] == "tree"]
test_event  = next((e for e in events if e["event"] == "test"), None)

DEPTH         = init_event["depth"]
TREE_COUNT    = init_event["tree_count"]
LEARNING_RATE = init_event["learning_rate"]
INIT_PRED     = init_event["initial_prediction"]

tree_indices      = [e["tree_index"]        for e in tree_events]
mean_residuals    = [e["mean_residual"]     for e in tree_events]
residual_variances= [e["residual_variance"] for e in tree_events]

# ── Helpers ───────────────────────────────────────────────────────────────────

def walk_tree(node, path="root"):
    """Yield (path, node) for every node depth-first."""
    yield path, node
    if node["type"] == "decision":
        yield from walk_tree(node["left"],  path + ".L")
        yield from walk_tree(node["right"], path + ".R")

def feature_usage(trees):
    """Count how many times each indicator is used as a split across all trees."""
    counter = Counter()
    for t in trees:
        for _, node in walk_tree(t):
            if node["type"] == "decision":
                counter[node["indicator"]] += 1
    return counter

def build_nx_graph(tree, tree_idx):
    """Convert a single tree dict into a labelled NetworkX DiGraph."""
    G = nx.DiGraph()
    def add_nodes(node, node_id):
        if node["type"] == "leaf":
            label = f"leaf\n{node['probability']:.4f}"
            G.add_node(node_id, label=label, color="#aed6f1")
        else:
            label = f"{node['indicator']}\n≤ {node['threshold']:.4f}"
            G.add_node(node_id, label=label, color="#a9dfbf")
            left_id  = node_id + "L"
            right_id = node_id + "R"
            G.add_edge(node_id, left_id,  label="≤")
            G.add_edge(node_id, right_id, label=">")
            add_nodes(node["left"],  left_id)
            add_nodes(node["right"], right_id)
    add_nodes(tree, f"T{tree_idx}_")
    return G

def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Compute top-down tree layout positions for a DiGraph."""
    pos = {}
    def _pos(node, left, right, vert):
        pos[node] = ((left + right) / 2, vert)
        children = list(G.successors(node))
        if children:
            dx = (right - left) / len(children)
            for i, child in enumerate(children):
                _pos(child, left + i * dx, left + (i + 1) * dx, vert - vert_gap)
    _pos(root, 0, width, vert_loc)
    return pos

# ── Figure 1 – Training convergence ──────────────────────────────────────────

fig1, axes = plt.subplots(1, 2, figsize=(13, 5))
fig1.suptitle(
    f"GBM Training Convergence  |  depth={DEPTH}  trees={TREE_COUNT}  lr={LEARNING_RATE}  init={INIT_PRED:.3f}",
    fontsize=12, fontweight="bold"
)

ax = axes[0]
ax.plot(tree_indices, mean_residuals, marker="o", color="#2980b9", linewidth=2)
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_title("Mean Residual After Each Tree")
ax.set_xlabel("Tree index")
ax.set_ylabel("Mean residual")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(tree_indices, residual_variances, marker="s", color="#e67e22", linewidth=2)
ax.set_title("Residual Variance After Each Tree")
ax.set_xlabel("Tree index")
ax.set_ylabel("Variance of residuals")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig(FOLDER+"convergence.png", dpi=150)
print("Saved convergence.png")

# ── Figure 2 – Feature importance (split frequency) ──────────────────────────

all_trees = [e["tree"] for e in tree_events]
usage = feature_usage(all_trees)

if usage:
    features, counts = zip(*sorted(usage.items(), key=lambda x: -x[1]))
    fig2, ax = plt.subplots(figsize=(max(8, len(features) * 0.7), 5))
    bars = ax.bar(features, counts, color="#8e44ad", edgecolor="white", linewidth=0.6)
    ax.bar_label(bars)
    ax.set_title("Feature Split Frequency Across All Trees", fontsize=12, fontweight="bold")
    ax.set_xlabel("Indicator")
    ax.set_ylabel("Number of splits")
    ax.set_xticklabels(features, rotation=40, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig2.savefig(FOLDER+"feature_importance.png", dpi=150)
    print("Saved feature_importance.png")

# ── Figure 3 – Test accuracy gauge ───────────────────────────────────────────

if test_event:
    correct   = test_event["correct"]
    incorrect = test_event["incorrect"]
    accuracy  = test_event["accuracy"]

    fig3, ax = plt.subplots(figsize=(5, 5))
    wedge_colors = ["#27ae60", "#e74c3c"]
    ax.pie(
        [correct, incorrect],
        labels=[f"Correct\n{correct}", f"Incorrect\n{incorrect}"],
        colors=wedge_colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    ax.set_title(f"Test Accuracy: {accuracy:.2f}%", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig3.savefig(FOLDER+"test_accuracy.png", dpi=150)
    print("Saved test_accuracy.png")

# ── Figure 4 – Tree structure diagrams ───────────────────────────────────────

n_trees = len(all_trees)
cols    = min(n_trees, 5)
rows    = (n_trees + cols - 1) // cols
fig4, axes4 = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
fig4.suptitle("Tree Structures (green = decision, blue = leaf)", fontsize=12, fontweight="bold")

# Flatten axes for uniform indexing
if n_trees == 1:
    axes4 = [[axes4]]
elif rows == 1:
    axes4 = [axes4]

for idx, tree in enumerate(all_trees):
    r, c = divmod(idx, cols)
    ax = axes4[r][c]
    G = build_nx_graph(tree, idx)
    root = f"T{idx}_"
    try:
        pos = hierarchy_pos(G, root, width=1.0, vert_gap=0.25)
        labels = nx.get_node_attributes(G, "label")
        colors = [G.nodes[n].get("color", "#cccccc") for n in G.nodes]
        nx.draw(
            G, pos, ax=ax,
            labels=labels,
            node_color=colors,
            node_size=1800,
            font_size=6,
            arrows=True,
            edge_color="#888888",
            width=1.2
        )
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    except Exception as e:
        ax.text(0.5, 0.5, f"(render error)\n{e}", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"Tree {idx}", fontsize=9)
    ax.axis("off")

# Hide unused subplots
for idx in range(n_trees, rows * cols):
    r, c = divmod(idx, cols)
    axes4[r][c].axis("off")

plt.tight_layout()
fig4.savefig(FOLDER+"tree_structures.png", dpi=150)
print("Saved tree_structures.png")

# ── Figure 5 – Per-tree leaf probability distribution ────────────────────────

fig5, ax5 = plt.subplots(figsize=(10, 5))
for idx, tree in enumerate(all_trees):
    leaf_probs = [node["probability"] for _, node in walk_tree(tree) if node["type"] == "leaf"]
    ax5.scatter([idx] * len(leaf_probs), leaf_probs, alpha=0.7, s=50)

ax5.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax5.set_title("Leaf Probability Values per Tree", fontsize=12, fontweight="bold")
ax5.set_xlabel("Tree index")
ax5.set_ylabel("Leaf probability")
ax5.grid(True, alpha=0.3)
plt.tight_layout()
fig5.savefig(FOLDER+"leaf_distributions.png", dpi=150)
print("Saved leaf_distributions.png")

print("\nAll done. Generated:")
print("  convergence.png        – mean residual & variance over training")
print("  feature_importance.png – split frequency per indicator")
print("  test_accuracy.png      – correct vs incorrect pie chart")
print("  tree_structures.png    – full diagram of every tree")
print("  leaf_distributions.png – leaf probability scatter per tree")