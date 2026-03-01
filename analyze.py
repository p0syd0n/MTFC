

import json
import os
import sys
import math
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np

BASE_FOLDER = "graphs/"
TREE_STRUCTURE=False

log_path = sys.argv[1] if len(sys.argv) > 1 else "training_log.json"

with open(log_path) as f:
    events = [json.loads(line) for line in f if line.strip()]

init_event     = next(e for e in events if e["event"] == "init")
tree_events    = [e for e in events if e["event"] == "tree"]
test_event     = next((e for e in events if e["event"] == "test"),     None)
backtest_event = next((e for e in events if e["event"] == "backtest"), None)

DEPTH         = init_event["depth"]
TREE_COUNT    = init_event["tree_count"]
LEARNING_RATE = init_event["learning_rate"]
INIT_PRED     = init_event["initial_prediction"]
MIN_LEAF_SIZE = init_event.get("min_leaf_size", 1)
TRAIN_FILE    = init_event.get("train_file", "unknown")
TEST_FILE     = init_event.get("test_file",  "unknown")

tree_indices       = [e["tree_index"]        for e in tree_events]
mean_residuals     = [e["mean_residual"]     for e in tree_events]
residual_variances = [e["residual_variance"] for e in tree_events]


lr_tag      = str(LEARNING_RATE).lstrip("0").replace(".", "")
folder_name = f"d{DEPTH}_t{TREE_COUNT}_lr{lr_tag}_ml{MIN_LEAF_SIZE}"
FOLDER      = os.path.join(BASE_FOLDER, folder_name, "")
os.makedirs(FOLDER, exist_ok=True)


hp = {
    "depth":          DEPTH,
    "tree_count":     TREE_COUNT,
    "learning_rate":  LEARNING_RATE,
    "min_leaf_size":  MIN_LEAF_SIZE,
    "initial_pred":   INIT_PRED,
    "train_file":     TRAIN_FILE,
    "test_file":      TEST_FILE,
    "log_file":       log_path,
    "output_folder":  FOLDER,
}
if test_event:
    hp["test_accuracy"]  = test_event["accuracy"]
    hp["test_correct"]   = test_event["correct"]
    hp["test_incorrect"] = test_event["incorrect"]

with open(FOLDER + "hyperparameters.json", "w") as hf:
    json.dump(hp, hf, indent=2)

print(f"Saving to : {FOLDER}")
print("Saved hyperparameters.json")


plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.9,
})


def _sci_label(x):
    """Self-contained scientific notation label for a single tick value."""
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    if -3 <= exp <= 3:
        return f"{x:.6g}"
    coef = x / (10 ** exp)
    return f"{coef:.3g}e{exp}"

def sci_formatter():
    return mticker.FuncFormatter(lambda x, _: _sci_label(x))

def apply_sci_if_needed(ax, axis="y"):
    """
    Apply independent scientific-notation tick labels when the axis range
    exponent falls outside [-3, 3].  Suppresses matplotlib's shared offset text
    so no scale indicator appears above/beside the axis.
    """
    lo, hi = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    mag = max(abs(lo), abs(hi))
    if mag == 0:
        return
    exp = math.floor(math.log10(mag))
    if exp > 3 or exp < -3:
        fmt_axis = ax.yaxis if axis == "y" else ax.xaxis
        fmt_axis.set_major_formatter(sci_formatter())
        fmt_axis.get_offset_text().set_visible(False)

def note(ax, text, loc="lower right"):
    """Small italic annotation placed inside the axes."""
    mapping = {
        "lower right": dict(xy=(0.99, 0.03), ha="right", va="bottom"),
        "lower left":  dict(xy=(0.01, 0.03), ha="left",  va="bottom"),
        "upper left":  dict(xy=(0.01, 0.97), ha="left",  va="top"),
        "upper right": dict(xy=(0.99, 0.97), ha="right", va="top"),
    }
    kw = mapping.get(loc, mapping["lower right"])
    ax.annotate(text, xycoords="axes fraction",
                fontsize=7.5, style="italic", color="#555555", **kw)


def walk_tree(node, path="root"):
    yield path, node
    if node["type"] == "decision":
        yield from walk_tree(node["left"],  path + ".L")
        yield from walk_tree(node["right"], path + ".R")

def feature_usage(trees):
    counter = Counter()
    for t in trees:
        for _, nd in walk_tree(t):
            if nd["type"] == "decision":
                counter[nd["indicator"]] += 1
    return counter

def build_nx_graph(tree, tree_idx):
    G = nx.DiGraph()
    def add_nodes(nd, nid):
        if nd["type"] == "leaf":
            G.add_node(nid, label=f"leaf\n{nd['probability']:.4f}", color="#aed6f1")
        else:
            G.add_node(nid, label=f"{nd['indicator']}\n≤{nd['threshold']:.4f}", color="#a9dfbf")
            lid, rid = nid + "L", nid + "R"
            G.add_edge(nid, lid, label="≤")
            G.add_edge(nid, rid, label=">")
            add_nodes(nd["left"],  lid)
            add_nodes(nd["right"], rid)
    add_nodes(tree, f"T{tree_idx}_")
    return G

def hierarchy_pos(G, root, width=1.0, vert_gap=0.2):
    pos = {}
    def _p(nd, lo, hi, v):
        pos[nd] = ((lo + hi) / 2, v)
        ch = list(G.successors(nd))
        if ch:
            dx = (hi - lo) / len(ch)
            for i, c in enumerate(ch):
                _p(c, lo + i * dx, lo + (i + 1) * dx, v - vert_gap)
    _p(root, 0, width, 0)
    return pos

all_trees = [e["tree"] for e in tree_events]
n_trees   = len(all_trees)

# Figure 1 — Training convergence

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13, 4.5))
fig1.suptitle("Training Convergence", fontsize=13, fontweight="bold")

ax1a.plot(tree_indices, mean_residuals, color="#2980b9", linewidth=1.5)
ax1a.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
ax1a.set_title("Mean Residual per Tree")
ax1a.set_xlabel("Tree index")
ax1a.set_ylabel("Mean residual")
apply_sci_if_needed(ax1a, "y")
note(ax1a, f"depth={DEPTH}  trees={TREE_COUNT}  lr={LEARNING_RATE}  min_leaf={MIN_LEAF_SIZE}")

ax1b.plot(tree_indices, residual_variances, color="#e67e22", linewidth=1.5)
ax1b.set_title("Residual Variance per Tree")
ax1b.set_xlabel("Tree index")
ax1b.set_ylabel("Variance")
apply_sci_if_needed(ax1b, "y")
note(ax1b, f"train: {TRAIN_FILE}  |  init pred = {INIT_PRED:.6f}")

plt.tight_layout()
fig1.savefig(FOLDER + "convergence.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved convergence.png")


usage = feature_usage(all_trees)
if usage:
    feats, cnts = zip(*sorted(usage.items(), key=lambda x: -x[1]))
    fig2, ax2 = plt.subplots(figsize=(max(9, len(feats) * 0.65), 5))
    fig2.suptitle("Feature Split Frequency", fontsize=13, fontweight="bold")
    bars = ax2.bar(feats, cnts, color="#4a6fa5", edgecolor="white", linewidth=0.5)
    ax2.bar_label(bars, fontsize=7.5, padding=2)
    ax2.set_xlabel("Feature")
    ax2.set_ylabel("Number of splits")
    ax2.set_xticklabels(feats, rotation=38, ha="right")
    note(ax2, f"{TREE_COUNT} trees  |  {sum(cnts)} total splits", "upper right")
    plt.tight_layout()
    fig2.savefig(FOLDER + "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved feature_importance.png")


if test_event:
    correct   = test_event["correct"]
    incorrect = test_event["incorrect"]
    accuracy  = test_event["accuracy"]

    fig3, ax3 = plt.subplots(figsize=(5, 5))
    fig3.suptitle("Test Set Accuracy", fontsize=13, fontweight="bold")
    ax3.pie(
        [correct, incorrect],
        labels=[f"Correct\n{correct:,}", f"Incorrect\n{incorrect:,}"],
        colors=["#27ae60", "#e74c3c"],
        autopct="%1.2f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    ax3.annotate(
        f"{accuracy:.4f}%  |  n = {correct + incorrect:,}  |  test: {TEST_FILE}",
        xy=(0.5, -0.02), xycoords="axes fraction",
        ha="center", fontsize=7.5, style="italic", color="#555555"
    )
    plt.tight_layout()
    fig3.savefig(FOLDER + "test_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Saved test_accuracy.png")

if TREE_STRUCTURE:
        
    cols  = min(n_trees, 5)
    rows  = (n_trees + cols - 1) // cols
    fig4, axes4 = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
    fig4.suptitle("Tree Structures  (green = decision,  blue = leaf)",
                fontsize=11, fontweight="bold")

    if n_trees == 1:
        axes4 = [[axes4]]
    elif rows == 1:
        axes4 = [list(axes4)]

    for idx, tree in enumerate(all_trees):
        r, c = divmod(idx, cols)
        ax   = axes4[r][c]
        G    = build_nx_graph(tree, idx)
        root = f"T{idx}_"
        try:
            pos    = hierarchy_pos(G, root)
            labels = nx.get_node_attributes(G, "label")
            colors = [G.nodes[nd].get("color", "#cccccc") for nd in G.nodes]
            nx.draw(G, pos, ax=ax, labels=labels, node_color=colors,
                    node_size=1800, font_size=6, arrows=True,
                    edge_color="#888888", width=1.2)
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=nx.get_edge_attributes(G, "label"),
                font_size=7, ax=ax)
        except Exception as e:
            ax.text(0.5, 0.5, f"render error\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        ax.set_title(f"Tree {idx}", fontsize=9)
        ax.axis("off")

    for idx in range(n_trees, rows * cols):
        r, c = divmod(idx, cols)
        axes4[r][c].axis("off")

    plt.tight_layout()
    fig4.savefig(FOLDER + "tree_structures.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("Saved tree_structures.png")



CANDLE_W  = 0.55
WHISKER_W = 0.22



fig5a, ax5a = plt.subplots(figsize=(max(14, n_trees * 0.20), 6))
fig5a.suptitle("Leaf Residuals — Scatter", fontsize=13, fontweight="bold")

for idx, tree in enumerate(all_trees):
    leaf_vals = [nd["probability"] for _, nd in walk_tree(tree)
                 if nd["type"] == "leaf"]
    ax5a.scatter([idx] * len(leaf_vals), leaf_vals, alpha=0.55, s=22, linewidths=0)

ax5a.axhline( 0, color="#888888", linestyle="--", linewidth=0.8)
ax5a.axhline( 1, color="#e74c3c", linestyle="--", linewidth=1.0,
              label="±1  (prediction outside [0, 1])")
ax5a.axhline(-1, color="#e74c3c", linestyle="--", linewidth=1.0)
ax5a.set_xlabel("Tree index")
ax5a.set_ylabel("Leaf residual")
ax5a.set_xlim(-1, n_trees)
apply_sci_if_needed(ax5a, "y")
ax5a.legend()
note(ax5a, "Each dot = one leaf.  Y = mean residual of training points routed to that leaf.")

plt.tight_layout()
fig5a.savefig(FOLDER + "leaf_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig5a)
print("Saved leaf_scatter.png")



fig5b, ax5b = plt.subplots(figsize=(max(14, n_trees * 0.22), 6))
fig5b.suptitle("Leaf Residuals — Candlestick (min/max wick, IQR body)",
               fontsize=13, fontweight="bold")

for idx, tree in enumerate(all_trees):
    vals = np.array(
        [nd["probability"] for _, nd in walk_tree(tree) if nd["type"] == "leaf"],
        dtype=float
    )
    if len(vals) == 0:
        continue

    v_min = vals.min()
    v_max = vals.max()
    q1    = np.percentile(vals, 25)
    med   = np.percentile(vals, 50)
    q3    = np.percentile(vals, 75)
    x     = idx
    col   = "#2980b9" if med >= 0 else "#c0392b"

    # Wick: absolute min → max
    ax5b.plot([x, x], [v_min, v_max], color=col, linewidth=0.9, zorder=2)
    ax5b.plot([x - WHISKER_W/2, x + WHISKER_W/2], [v_min, v_min],
              color=col, linewidth=0.9, zorder=2)
    ax5b.plot([x - WHISKER_W/2, x + WHISKER_W/2], [v_max, v_max],
              color=col, linewidth=0.9, zorder=2)

    # IQR body
    body_h = max(q3 - q1, 1e-6)
    rect = mpatches.FancyBboxPatch(
        (x - CANDLE_W/2, q1), CANDLE_W, body_h,
        boxstyle="square,pad=0",
        facecolor=col, alpha=0.65, edgecolor=col, linewidth=0.7, zorder=3
    )
    ax5b.add_patch(rect)

    # Median bar
    ax5b.plot([x - CANDLE_W/2, x + CANDLE_W/2], [med, med],
              color="white", linewidth=1.5, zorder=4)

ax5b.axhline( 0, color="#888888", linestyle="--", linewidth=0.8, zorder=1)
ax5b.axhline( 1, color="#e74c3c", linestyle="--", linewidth=1.0, zorder=1)
ax5b.axhline(-1, color="#e74c3c", linestyle="--", linewidth=1.0, zorder=1)

ax5b.legend(handles=[
    mpatches.Patch(facecolor="#2980b9", alpha=0.7, label="Median ≥ 0"),
    mpatches.Patch(facecolor="#c0392b", alpha=0.7, label="Median < 0"),
    plt.Line2D([0], [0], color="white", linewidth=2, label="Median (white bar)"),
    plt.Line2D([0], [0], color="#e74c3c", linestyle="--", linewidth=1.0,
               label="±1  (prediction outside [0, 1])"),
], loc="upper right")
ax5b.set_xlabel("Tree index")
ax5b.set_ylabel("Leaf residual")
ax5b.set_xlim(-1, n_trees)
apply_sci_if_needed(ax5b, "y")
note(ax5b, "Wick = absolute min / max  |  Body = IQR (Q1–Q3)  |  White bar = median")

plt.tight_layout()
fig5b.savefig(FOLDER + "leaf_candlestick.png", dpi=150, bbox_inches="tight")
plt.close(fig5b)
print("Saved leaf_candlestick.png")



fig5c, ax5c = plt.subplots(figsize=(max(14, n_trees * 0.22), 6))
fig5c.suptitle("Leaf Residuals — Candlestick (Tukey whiskers, outlier dots)",
               fontsize=13, fontweight="bold")

for idx, tree in enumerate(all_trees):
    vals = np.array(
        [nd["probability"] for _, nd in walk_tree(tree) if nd["type"] == "leaf"],
        dtype=float
    )
    if len(vals) == 0:
        continue

    q1   = np.percentile(vals, 25)
    med  = np.percentile(vals, 50)
    q3   = np.percentile(vals, 75)
    iqr  = q3 - q1
    lo_f = q1 - 1.5 * iqr
    hi_f = q3 + 1.5 * iqr
    w_lo = vals[vals >= lo_f].min() if (vals >= lo_f).any() else q1
    w_hi = vals[vals <= hi_f].max() if (vals <= hi_f).any() else q3
    outs = vals[(vals < lo_f) | (vals > hi_f)]
    x    = idx
    col  = "#2980b9" if med >= 0 else "#c0392b"

    # Whiskers to Tukey fences
    ax5c.plot([x, x], [w_lo, q1], color=col, linewidth=0.9, zorder=2)
    ax5c.plot([x, x], [q3, w_hi], color=col, linewidth=0.9, zorder=2)
    ax5c.plot([x - WHISKER_W/2, x + WHISKER_W/2], [w_lo, w_lo],
              color=col, linewidth=0.9, zorder=2)
    ax5c.plot([x - WHISKER_W/2, x + WHISKER_W/2], [w_hi, w_hi],
              color=col, linewidth=0.9, zorder=2)

    # IQR body
    body_h = max(q3 - q1, 1e-6)
    rect = mpatches.FancyBboxPatch(
        (x - CANDLE_W/2, q1), CANDLE_W, body_h,
        boxstyle="square,pad=0",
        facecolor=col, alpha=0.65, edgecolor=col, linewidth=0.7, zorder=3
    )
    ax5c.add_patch(rect)

    # Median bar
    ax5c.plot([x - CANDLE_W/2, x + CANDLE_W/2], [med, med],
              color="white", linewidth=1.5, zorder=4)

    # Outlier dots
    if len(outs) > 0:
        ax5c.scatter([x] * len(outs), outs,
                     marker="o", s=14, color="#e67e22", alpha=0.85, zorder=5)

ax5c.axhline( 0, color="#888888", linestyle="--", linewidth=0.8, zorder=1)
ax5c.axhline( 1, color="#e74c3c", linestyle="--", linewidth=1.0, zorder=1)
ax5c.axhline(-1, color="#e74c3c", linestyle="--", linewidth=1.0, zorder=1)

ax5c.legend(handles=[
    mpatches.Patch(facecolor="#2980b9", alpha=0.7, label="Median ≥ 0"),
    mpatches.Patch(facecolor="#c0392b", alpha=0.7, label="Median < 0"),
    plt.Line2D([0], [0], color="white", linewidth=2, label="Median (white bar)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22",
               markersize=5, label="Outlier leaf (beyond 1.5×IQR)"),
    plt.Line2D([0], [0], color="#e74c3c", linestyle="--", linewidth=1.0,
               label="±1  (prediction outside [0, 1])"),
], loc="upper right")
ax5c.set_xlabel("Tree index")
ax5c.set_ylabel("Leaf residual")
ax5c.set_xlim(-1, n_trees)
apply_sci_if_needed(ax5c, "y")
note(ax5c, "Whiskers = 1.5×IQR fence  |  Body = IQR  |  White bar = median  |  Dots = outliers")

plt.tight_layout()
fig5c.savefig(FOLDER + "leaf_candlestick_tukey.png", dpi=150, bbox_inches="tight")
plt.close(fig5c)
print("Saved leaf_candlestick_tukey.png")


if backtest_event:
    points      = backtest_event["points"]
    preds       = np.array([p["p"] for p in points], dtype=float)
    labels      = np.array([p["l"] for p in points], dtype=int)
    n           = len(preds)
    correct_arr = ((preds > 0.5) == labels.astype(bool)).astype(int)
    preds_c     = np.clip(preds, 0.0, 1.0)
    n_out_lo    = int((preds < 0.0).sum())
    n_out_hi    = int((preds > 1.0).sum())
    trades      = np.arange(n)

    pnl      = np.where(correct_arr, 1, -1).cumsum()
    peak     = np.maximum.accumulate(pnl)
    drawdown = pnl - peak

    # Load test file for OHLC overlay if available
    ohlc_data = None
    if not TEST_FILE or TEST_FILE == "unknown":
        print("OHLC overlay skipped: no test file recorded in training log.")
    elif not os.path.exists(TEST_FILE):
        print(f"OHLC overlay skipped: test file not found at '{TEST_FILE}'.")
    else:
        import csv
        with open(TEST_FILE) as tf:
            reader = csv.DictReader(tf)
            raw_rows = list(reader)
        if not raw_rows:
            print(f"OHLC overlay skipped: test file '{TEST_FILE}' is empty.")
        else:
            raw_rows = raw_rows[:n]  # test file has one extra bar (no next-bar for last row)
            try:
                # Normalise column names: strip whitespace and lowercase
                norm_keys = {k.strip().lower(): k for k in raw_rows[0].keys()}
                o_key = norm_keys.get("open")
                h_key = norm_keys.get("high")
                l_key = norm_keys.get("low")
                c_key = norm_keys.get("close")
                missing = [name for name, key in [("open", o_key), ("high", h_key), ("low", l_key), ("close", c_key)] if key is None]
                if missing:
                    print(f"OHLC overlay skipped: column(s) not found in test file: {missing}. Available columns: {list(norm_keys.keys())}")
                else:
                    ohlc_data = {
                        "open":  np.array([float(r[o_key]) for r in raw_rows]),
                        "high":  np.array([float(r[h_key]) for r in raw_rows]),
                        "low":   np.array([float(r[l_key]) for r in raw_rows]),
                        "close": np.array([float(r[c_key]) for r in raw_rows]),
                    }
            except Exception as e:
                print(f"OHLC overlay skipped: could not parse OHLC columns: {e}")

    #
    # Top panel (3/4 height):
    #   Left axis  — cumulative P&L
    #   Right axis — OHLC candlestick price per trade (if available),
    #                otherwise close line
    #
    # Bottom panel (1/4 height):
    #   Drawdown

    fig6, (ax6a, ax6b) = plt.subplots(
        2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig6.suptitle("Backtest Equity Curve", fontsize=13, fontweight="bold")

    # P&L — left axis
    ax6a.plot(trades, pnl, color="#2980b9", linewidth=1.4, label="Cumulative P&L")
    ax6a.fill_between(trades, pnl, alpha=0.08, color="#2980b9")
    ax6a.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax6a.set_ylabel("Cumulative P&L (units)")
    ax6a.set_xlim(0, n - 1)
    apply_sci_if_needed(ax6a, "y")
    note(ax6a, f"Final P&L = {pnl[-1]:+,}  |  n = {n:,}  |  test: {TEST_FILE}")

    # OHLC — right axis
    if ohlc_data is not None:
        ax6a_price = ax6a.twinx()
        ax6a_price.spines["right"].set_visible(True)

        o = ohlc_data["open"]
        h = ohlc_data["high"]
        l = ohlc_data["low"]
        c = ohlc_data["close"]

        # Draw individual OHLC candlesticks only when n is small enough to be legible;
        # otherwise fall back to high/low band + close line for readability.
        if n <= 2000:
            candle_w = max(0.3, min(0.8, 300 / n))
            for i in trades:
                col_c = "#27ae60" if c[i] >= o[i] else "#c0392b"
                # High-low wick
                ax6a_price.plot([i, i], [l[i], h[i]],
                                color=col_c, linewidth=0.7, zorder=2)
                # Open-close body
                body_lo = min(o[i], c[i])
                body_hi = max(o[i], c[i])
                rect = mpatches.FancyBboxPatch(
                    (i - candle_w / 2, body_lo), candle_w,
                    max(body_hi - body_lo, (h[i] - l[i]) * 0.01),
                    boxstyle="square,pad=0",
                    facecolor=col_c, edgecolor=col_c,
                    linewidth=0.4, alpha=0.75, zorder=3
                )
                ax6a_price.add_patch(rect)
            price_label = "OHLC (green=up, red=down)"
        else:
            # High/low shaded band + close line
            ax6a_price.fill_between(trades, l, h,
                                    alpha=0.18, color="#888888", label="High/Low band")
            ax6a_price.plot(trades, c, color="#555555", linewidth=0.7,
                            alpha=0.80, label="Close price")
            price_label = "Price (band=H/L, line=close)"

        ax6a_price.set_ylabel("Price")
        apply_sci_if_needed(ax6a_price, "y")

        lines_l, labels_l = ax6a.get_legend_handles_labels()
        lines_p = [mpatches.Patch(facecolor="#27ae60", alpha=0.7, label=price_label)]
        ax6a.legend(lines_l + lines_p, labels_l + [price_label], loc="upper left", ncol=2)
    else:
        lines_l, labels_l = ax6a.get_legend_handles_labels()
        ax6a.legend(lines_l, labels_l, loc="upper left")

    # Drawdown panel
    ax6b.fill_between(trades, drawdown, color="#e74c3c", alpha=0.7, label="Drawdown")
    ax6b.set_ylabel("Drawdown (units)")
    ax6b.set_xlabel("Trade index")
    ax6b.set_xlim(0, n - 1)
    ax6b.legend(loc="lower left")
    apply_sci_if_needed(ax6b, "y")
    note(ax6b, f"Max drawdown = {drawdown.min():,}")

    plt.tight_layout()
    fig6.savefig(FOLDER + "backtest_equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig6)
    print("Saved backtest_equity_curve.png")

    #
    # Left axis  — rolling accuracy (%)
    # Right axis — rolling mean close price over the same window (if available)

    WINDOW       = max(50, n // 40)
    rolling_acc  = np.convolve(correct_arr, np.ones(WINDOW) / WINDOW, mode="valid") * 100
    x_roll       = np.arange(WINDOW - 1, n)
    mean_acc     = rolling_acc.mean()

    fig7, ax7 = plt.subplots(figsize=(16, 5))
    fig7.suptitle("Rolling Accuracy", fontsize=13, fontweight="bold")

    # Accuracy — left axis
    ax7.plot(x_roll, rolling_acc, color="#27ae60", linewidth=1.4, label="Accuracy (%)")
    ax7.axhline(50,       color="#888888", linestyle="--", linewidth=0.8, label="50%")
    ax7.axhline(mean_acc, color="#27ae60", linestyle="-.",
                linewidth=1.0, label=f"Mean {mean_acc:.2f}%")
    ax7.fill_between(x_roll, rolling_acc, 50,
                     where=(rolling_acc >= 50), alpha=0.08, color="#27ae60")
    ax7.fill_between(x_roll, rolling_acc, 50,
                     where=(rolling_acc  < 50), alpha=0.08, color="#e74c3c")
    lo_y = max(20, rolling_acc.min() - 5)
    hi_y = min(100, rolling_acc.max() + 5)
    ax7.set_ylim(lo_y, hi_y)
    ax7.set_ylabel("Accuracy (%)")
    ax7.set_xlabel("Period index")
    note(ax7, f"Window = {WINDOW} periods  |  test: {TEST_FILE}")

    # Rolling mean close price — right axis (if OHLC available)
    if ohlc_data is not None:
        roll_close = np.convolve(ohlc_data["close"], np.ones(WINDOW) / WINDOW, mode="valid")
        ax7_price  = ax7.twinx()
        ax7_price.spines["right"].set_visible(True)
        ax7_price.plot(x_roll, roll_close, color="#555555", linewidth=1.0,
                       alpha=0.75, label=f"Rolling mean close (w={WINDOW})")
        ax7_price.set_ylabel(f"Rolling mean close price (w={WINDOW})")
        apply_sci_if_needed(ax7_price, "y")
        lines_p, labels_p = ax7_price.get_legend_handles_labels()
    else:
        lines_p, labels_p = [], []

    lines_l, labels_l = ax7.get_legend_handles_labels()
    ax7.legend(lines_l + lines_p, labels_l + labels_p, loc="upper left", ncol=2)

    plt.tight_layout()
    fig7.savefig(FOLDER + "backtest_rolling_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig7)
    print("Saved backtest_rolling_accuracy.png")


    fig8, ax8 = plt.subplots(figsize=(10, 5))
    fig8.suptitle("Prediction Score Distribution by True Outcome",
                  fontsize=13, fontweight="bold")
    bins = np.linspace(0, 1, 41)
    ax8.hist(preds_c[labels == 1], bins=bins, alpha=0.6, color="#27ae60",
             label="True outcome = BUY",  edgecolor="white")
    ax8.hist(preds_c[labels == 0], bins=bins, alpha=0.6, color="#e74c3c",
             label="True outcome = SELL", edgecolor="white")
    ax8.axvline(0.5, color="black", linestyle="--", linewidth=1.2,
                label="Decision boundary (0.5)")
    ax8.set_xlabel("Prediction score")
    ax8.set_ylabel("Count")
    ax8.legend()
    out_note = ""
    if n_out_lo > 0 or n_out_hi > 0:
        out_note = f"  |  {n_out_lo} clipped below 0,  {n_out_hi} clipped above 1"
    note(ax8, f"Scores clipped to [0, 1] for display{out_note}")
    plt.tight_layout()
    fig8.savefig(FOLDER + "backtest_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig8)
    print("Saved backtest_score_distribution.png")


    N_BINS    = 10
    bin_edges = np.linspace(0, 1, N_BINS + 1)
    bin_pred_vals, bin_actual_vals, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (preds_c >= lo) & (preds_c < hi)
        if mask.sum() > 0:
            bin_pred_vals.append(preds_c[mask].mean())
            bin_actual_vals.append(labels[mask].mean())
            bin_counts.append(int(mask.sum()))
    bin_pred_arr   = np.array(bin_pred_vals)
    bin_actual_arr = np.array(bin_actual_vals)
    bin_counts_arr = np.array(bin_counts)

    fig9, ax9 = plt.subplots(figsize=(7, 7))
    fig9.suptitle("Calibration", fontsize=13, fontweight="bold")
    ax9.plot([0, 1], [0, 1], color="#888888", linestyle="--",
             linewidth=1.2, label="Perfect calibration", zorder=1)
    sc = ax9.scatter(bin_pred_arr, bin_actual_arr, c=bin_counts_arr, cmap="RdYlGn",
                     s=220, zorder=3, edgecolors="#333333", linewidths=0.8,
                     vmin=0, vmax=bin_counts_arr.max())
    ax9.plot(bin_pred_arr, bin_actual_arr, color="#2980b9", linewidth=1.4,
             alpha=0.6, zorder=2)
    cbar = plt.colorbar(sc, ax=ax9, fraction=0.046, pad=0.04)
    cbar.set_label("Samples in bin", fontsize=9)
    for xp, ya, cnt in zip(bin_pred_arr, bin_actual_arr, bin_counts_arr):
        ha = "left"  if xp < 0.80 else "right"
        ox = 0.014   if xp < 0.80 else -0.014
        ax9.text(xp + ox, ya + 0.022, f"n={cnt:,}",
                 fontsize=8, fontweight="bold", ha=ha, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                           edgecolor="#cccccc", alpha=0.92))
    ax9.set_xlabel("Mean predicted score")
    ax9.set_ylabel("Actual win rate")
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.legend()
    n_out_total = n_out_lo + n_out_hi
    note(ax9, (f"{n_out_total} predictions outside [0, 1] excluded"
               if n_out_total else "All predictions within [0, 1]"))
    plt.tight_layout()
    fig9.savefig(FOLDER + "backtest_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig9)
    print("Saved backtest_calibration.png")


    confidence  = np.abs(preds_c - 0.5)
    N_CONF      = 8
    conf_edges  = np.unique(np.percentile(confidence, np.linspace(0, 100, N_CONF + 1)))
    conf_mid, conf_acc, conf_cnt = [], [], []
    for lo, hi in zip(conf_edges[:-1], conf_edges[1:]):
        mask = (confidence >= lo) & (confidence <= hi)
        if mask.sum() > 0:
            conf_mid.append(confidence[mask].mean())
            conf_acc.append(correct_arr[mask].mean() * 100)
            conf_cnt.append(int(mask.sum()))
    conf_mid = np.array(conf_mid)
    conf_acc = np.array(conf_acc)
    conf_cnt = np.array(conf_cnt)

    fig10, ax10 = plt.subplots(figsize=(9, 5))
    fig10.suptitle("Confidence vs Accuracy", fontsize=13, fontweight="bold")
    bar_cols = ["#27ae60" if a >= 50 else "#e74c3c" for a in conf_acc]
    bars10 = ax10.bar(range(len(conf_mid)), conf_acc, color=bar_cols,
                      edgecolor="white", linewidth=0.6)
    ax10.bar_label(bars10, fmt="%.2f%%", padding=3, fontsize=8)
    ax10.set_xticks(range(len(conf_mid)))
    ax10.set_xticklabels([f"{v:.4f}" for v in conf_mid], rotation=30, ha="right")
    ax10.axhline(50, color="#888888", linestyle="--", linewidth=1.0)
    ax10.set_xlabel("|prediction − 0.5|  (higher = more confident)")
    ax10.set_ylabel("Accuracy (%)")
    ylo = max(30, conf_acc.min() - 8)
    yhi = min(100, conf_acc.max() + 10)
    ax10.set_ylim(ylo, yhi)
    for i, cnt in enumerate(conf_cnt):
        ax10.text(i, ylo + 1.0, f"n={cnt:,}", ha="center", fontsize=7, color="#444444")
    plt.tight_layout()
    fig10.savefig(FOLDER + "backtest_confidence_vs_accuracy.png",
                  dpi=150, bbox_inches="tight")
    plt.close(fig10)
    print("Saved backtest_confidence_vs_accuracy.png")


    buy_mask  = preds_c > 0.5
    sell_mask = ~buy_mask
    buy_acc   = correct_arr[buy_mask].mean()  * 100 if buy_mask.sum()  > 0 else 0.0
    sell_acc  = correct_arr[sell_mask].mean() * 100 if sell_mask.sum() > 0 else 0.0

    fig11, axes11 = plt.subplots(1, 2, figsize=(10, 5))
    fig11.suptitle("Accuracy by Signal Direction", fontsize=13, fontweight="bold")
    for ax_, lbl_, acc_, cnt_, col_ in [
        (axes11[0], "BUY  (pred > 0.5)",  buy_acc,  buy_mask.sum(),  "#27ae60"),
        (axes11[1], "SELL (pred ≤ 0.5)", sell_acc, sell_mask.sum(), "#e74c3c"),
    ]:
        ax_.pie(
            [acc_, 100 - acc_],
            labels=[f"Correct\n{acc_:.2f}%", f"Incorrect\n{100-acc_:.2f}%"],
            colors=[col_, "#cccccc"],
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            autopct="%1.2f%%",
        )
        ax_.set_title(lbl_, fontsize=10)
        ax_.annotate(f"n = {cnt_:,}", xy=(0.5, -0.03), xycoords="axes fraction",
                     ha="center", fontsize=8, style="italic", color="#555555")
    plt.tight_layout()
    fig11.savefig(FOLDER + "backtest_long_short_split.png",
                  dpi=150, bbox_inches="tight")
    plt.close(fig11)
    print("Saved backtest_long_short_split.png")

    print(f"\nBacktest summary  (n = {n:,})")
    print(f"  BUY  signals : {buy_mask.sum():>6,}   accuracy : {buy_acc:.2f}%")
    print(f"  SELL signals : {sell_mask.sum():>6,}   accuracy : {sell_acc:.2f}%")
    print(f"  Outside [0,1]: {n_out_lo} below 0,  {n_out_hi} above 1")
    print(f"  Max drawdown : {drawdown.min():,}")
    print(f"  Final P&L    : {pnl[-1]:+,}")

print(f"\nAll charts saved to: {FOLDER}")