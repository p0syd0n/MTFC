import json
import os
import sys
import math
import csv
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np

# Global Config
BASE_DIR = "graphs/"
SHOW_TREE_MAPS = False  # Toggle for tree structure diagrams
CANDLE_WIDTH = 0.55
WICK_WIDTH = 0.22

# Setup paths
log_file = sys.argv[1] if len(sys.argv) > 1 else "training_log.json"

if not os.path.exists(log_file):
    print(f"Error: {log_path} not found.")
    sys.exit(1)

# Parsing events
with open(log_file) as f:
    events = [json.loads(line) for line in f if line.strip()]

init_ev      = next(e for e in events if e["event"] == "init")
tree_evs     = [e for e in events if e["event"] == "tree"]
test_ev      = next((e for e in events if e["event"] == "test"), None)
backtest_ev  = next((e for e in events if e["event"] == "backtest"), None)

# Hyperparams
DEPTH    = init_ev["depth"]
TREES    = init_ev["tree_count"]
LR       = init_ev["learning_rate"]
START_P  = init_ev["initial_prediction"]
MIN_LEAF = init_ev.get("min_leaf_size", 1)
TR_FILE  = init_ev.get("train_file", "unknown")
TS_FILE  = init_ev.get("test_file",  "unknown")

# Dir management
lr_str = str(LR).lstrip("0").replace(".", "")
out_dir = os.path.join(BASE_DIR, f"d{DEPTH}_t{TREES}_lr{lr_str}_ml{MIN_LEAF}", "")
os.makedirs(out_dir, exist_ok=True)

# Export metadata
meta = {
    "depth": DEPTH, "tree_count": TREES, "learning_rate": LR,
    "min_leaf_size": MIN_LEAF, "initial_pred": START_P,
    "train_file": TR_FILE, "test_file": TS_FILE, "log_file": log_file,
    "output_folder": out_dir
}
if test_ev:
    meta.update({"test_accuracy": test_ev["accuracy"], "test_correct": test_ev["correct"], "test_incorrect": test_ev["incorrect"]})

with open(out_dir + "hyperparameters.json", "w") as hf:
    json.dump(meta, hf, indent=2)

# --- Plot Styles ---
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linestyle": "--",
    "axes.titlesize": 11, "legend.fontsize": 8
})

def format_sci(x):
    if x == 0: return "0"
    exp = int(math.floor(math.log10(abs(x))))
    if -3 <= exp <= 3: return f"{x:.6g}"
    return f"{x / (10**exp):.3g}e{exp}"

def fix_axis_scaling(ax, axis="y"):
    lims = ax.get_ylim() if axis == "y" else ax.get_xlim()
    if max(abs(lims[0]), abs(lims[1])) > 1000 or max(abs(lims[0]), abs(lims[1])) < 0.001:
        f = ax.yaxis if axis == "y" else ax.xaxis
        f.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_sci(x)))
        f.get_offset_text().set_visible(False)

def add_note(ax, txt, loc="lower right"):
    pos = {
        "lower right": (0.99, 0.03, "right", "bottom"),
        "lower left":  (0.01, 0.03, "left", "bottom"),
        "upper left":  (0.01, 0.97, "left", "top"),
        "upper right": (0.99, 0.97, "right", "top"),
    }.get(loc, (0.99, 0.03, "right", "bottom"))
    ax.annotate(txt, xy=(pos[0], pos[1]), xycoords="axes fraction",
                ha=pos[2], va=pos[3], fontsize=7.5, style="italic", color="#555555")

# --- Logic ---
def get_tree_nodes(node, path="root"):
    yield path, node
    if node["type"] == "decision":
        yield from get_tree_nodes(node["left"], path + ".L")
        yield from get_tree_nodes(node["right"], path + ".R")

def get_splits(trees):
    c = Counter()
    for t in trees:
        for _, n in get_tree_nodes(t):
            if n["type"] == "decision": c[n["indicator"]] += 1
    return c

all_t = [e["tree"] for e in tree_evs]

# Fig 1: Convergence
f1, (a1a, a1b) = plt.subplots(1, 2, figsize=(13, 4.5))
f1.suptitle("Training Convergence", fontweight="bold")
idx = [e["tree_index"] for e in tree_evs]
a1a.plot(idx, [e["mean_residual"] for e in tree_evs], color="#2980b9")
a1a.axhline(0, color="#888888", ls="--", lw=0.8)
a1a.set_title("Mean Residual per Tree")
fix_axis_scaling(a1a)
add_note(a1a, f"depth={DEPTH} trees={TREES} lr={LR}")

a1b.plot(idx, [e["residual_variance"] for e in tree_evs], color="#e67e22")
a1b.set_title("Residual Variance per Tree")
fix_axis_scaling(a1b)
add_note(a1b, f"init pred: {START_P:.6f}")
f1.savefig(out_dir + "convergence.png", dpi=150); plt.close(f1)

# Fig 2: Importance
usage = get_splits(all_t)
if usage:
    f, c = zip(*sorted(usage.items(), key=lambda x: -x[1]))
    f2, ax2 = plt.subplots(figsize=(max(9, len(f)*0.65), 5))
    bars = ax2.bar(f, c, color="#4a6fa5")
    ax2.bar_label(bars, fontsize=7.5, padding=2)
    ax2.set_xticklabels(f, rotation=38, ha="right")
    ax2.set_title("Feature Split Frequency", fontweight="bold")
    f2.savefig(out_dir + "feature_importance.png", dpi=150); plt.close(f2)

# Fig 3: Test Acc
if test_ev:
    f3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.pie([test_ev["correct"], test_ev["incorrect"]], 
           labels=[f"Correct\n{test_ev['correct']:,}", f"Incorrect\n{test_ev['incorrect']:,}"],
           colors=["#27ae60", "#e74c3c"], autopct="%1.2f%%", startangle=90)
    ax3.set_title("Test Set Accuracy", fontweight="bold")
    f3.savefig(out_dir + "test_accuracy.png", dpi=150); plt.close(f3)

# Fig 4: Topology
if SHOW_TREE_MAPS:
    cols = min(len(all_t), 5)
    rows = (len(all_t) + cols - 1) // cols
    f4, axes = plt.subplots(rows, cols, figsize=(cols*4.5, rows*4))
    if len(all_t) == 1: axes = [[axes]]
    elif rows == 1: axes = [list(axes)]
    
    for i, t in enumerate(all_t):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        G = nx.DiGraph()
        def build(nd, nid):
            if nd["type"] == "leaf":
                G.add_node(nid, label=f"leaf\n{nd['probability']:.4f}", color="#aed6f1")
            else:
                G.add_node(nid, label=f"{nd['indicator']}\n≤{nd['threshold']:.4f}", color="#a9dfbf")
                G.add_edge(nid, nid+"L", label="≤"); G.add_edge(nid, nid+"R", label=">")
                build(nd["left"], nid+"L"); build(nd["right"], nid+"R")
        
        build(t, f"T{i}_")
        try:
            # simple layout logic
            pos = {}; 
            def _pos(nd, lo, hi, v):
                pos[nd] = ((lo+hi)/2, v)
                subs = list(G.successors(nd))
                if subs:
                    dx = (hi-lo)/len(subs)
                    for k, child in enumerate(subs): _pos(child, lo+k*dx, lo+(k+1)*dx, v-0.2)
            _pos(f"T{i}_", 0, 1.0, 0)
            
            nx.draw(G, pos, ax=ax, labels=nx.get_node_attributes(G, "label"), 
                    node_color=[G.nodes[n]["color"] for n in G.nodes],
                    node_size=1800, font_size=6, arrows=True)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"), font_size=7, ax=ax)
        except: pass
        ax.set_title(f"Tree {i}")
        ax.axis("off")
    f4.savefig(out_dir + "tree_structures.png", dpi=150); plt.close(f4)

# Fig 5: Residual distribution
for suffix, mode in [("scatter", 0), ("candle", 1), ("tukey", 2)]:
    fig, ax = plt.subplots(figsize=(max(14, len(all_t)*0.22), 6))
    for idx, t in enumerate(all_t):
        vals = np.array([n["probability"] for _, n in get_tree_nodes(t) if n["type"] == "leaf"])
        if len(vals) == 0: continue
        
        x = idx
        col = "#2980b9" if np.median(vals) >= 0 else "#c0392b"
        
        if mode == 0:
            ax.scatter([x]*len(vals), vals, alpha=0.55, s=22, lw=0)
        elif mode == 1:
            ax.plot([x,x], [vals.min(), vals.max()], color=col, lw=0.9)
            q1, q3 = np.percentile(vals, [25, 75])
            ax.add_patch(mpatches.Rectangle((x-CANDLE_WIDTH/2, q1), CANDLE_WIDTH, max(q3-q1, 1e-6), fc=col, alpha=0.65))
            ax.plot([x-CANDLE_WIDTH/2, x+CANDLE_WIDTH/2], [np.median(vals)]*2, color="white", lw=1.5)
        elif mode == 2:
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            w_lo = vals[vals >= q1 - 1.5*iqr].min() if any(vals >= q1-1.5*iqr) else q1
            w_hi = vals[vals <= q3 + 1.5*iqr].max() if any(vals <= q3+1.5*iqr) else q3
            ax.plot([x,x], [w_lo, q1], color=col); ax.plot([x,x], [q3, w_hi], color=col)
            ax.add_patch(mpatches.Rectangle((x-CANDLE_WIDTH/2, q1), CANDLE_WIDTH, max(q3-q1, 1e-6), fc=col, alpha=0.65))
            ax.plot([x-CANDLE_WIDTH/2, x+CANDLE_WIDTH/2], [med]*2, color="white", lw=1.5)
            outs = vals[(vals < q1 - 1.5*iqr) | (vals > q3 + 1.5*iqr)]
            if len(outs): ax.scatter([x]*len(outs), outs, s=14, color="#e67e22")

    ax.axhline(0, color="#888888", ls="--", lw=0.8)
    ax.axhline(1, color="#e74c3c", ls="--", lw=1.0); ax.axhline(-1, color="#e74c3c", ls="--", lw=1.0)
    ax.set_title(f"Leaf Residuals ({suffix})", fontweight="bold")
    fig.savefig(f"{out_dir}leaf_{suffix if mode != 2 else 'candlestick_tukey'}.png", dpi=150); plt.close(fig)

# Backtesting
if backtest_ev:
    pts = backtest_ev["points"]
    preds = np.array([p["p"] for p in pts])
    lbls  = np.array([p["l"] for p in pts])
    n_pts = len(preds)
    correct = ((preds > 0.5) == lbls.astype(bool)).astype(int)
    
    pnl = np.where(correct, 1, -1).cumsum()
    dd  = pnl - np.maximum.accumulate(pnl)

    # OHLC Loader
    ohlc = None
    if TS_FILE and TS_FILE != "unknown" and os.path.exists(TS_FILE):
        with open(TS_FILE) as tf:
            r = list(csv.DictReader(tf))[:n_pts]
            keys = {k.strip().lower(): k for k in r[0].keys()}
            try:
                ohlc = {k: np.array([float(row[keys[k]]) for row in r]) for k in ['open','high','low','close']}
            except: pass

    # Fig 6: Equity
    f6, (axA, axB) = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]})
    axA.plot(pnl, color="#2980b9", label="Cumulative P&L")
    axA.fill_between(range(n_pts), pnl, alpha=0.08, color="#2980b9")
    
    if ohlc:
        px = axA.twinx()
        if n_pts <= 2000:
            for i in range(n_pts):
                c_ = "#27ae60" if ohlc['close'][i] >= ohlc['open'][i] else "#c0392b"
                px.plot([i,i], [ohlc['low'][i], ohlc['high'][i]], color=c_, lw=0.7)
                px.add_patch(mpatches.Rectangle((i-0.3, min(ohlc['open'][i], ohlc['close'][i])), 0.6, max(abs(ohlc['open'][i]-ohlc['close'][i]), 1e-5), fc=c_, alpha=0.7))
        else:
            px.fill_between(range(n_pts), ohlc['low'], ohlc['high'], alpha=0.1, color="#888888")
            px.plot(ohlc['close'], color="#555555", lw=0.7, alpha=0.6)
    
    axB.fill_between(range(n_pts), dd, color="#e74c3c", alpha=0.7)
    f6.savefig(out_dir + "backtest_equity_curve.png", dpi=150); plt.close(f6)

    # Fig 7: Rolling Acc
    win = max(50, n_pts // 40)
    roll_acc = np.convolve(correct, np.ones(win)/win, mode='valid') * 100
    f7, ax7 = plt.subplots(figsize=(16, 5))
    ax7.plot(range(win-1, n_pts), roll_acc, color="#27ae60")
    ax7.axhline(50, color="#888888", ls="--")
    if ohlc:
        ax7p = ax7.twinx()
        ax7p.plot(range(win-1, n_pts), np.convolve(ohlc['close'], np.ones(win)/win, mode='valid'), color="#555555", alpha=0.4)
    f7.savefig(out_dir + "backtest_rolling_accuracy.png", dpi=150); plt.close(f7)

    # Fig 8 & 9: Scores/Calibration
    f8, ax8 = plt.subplots(figsize=(10, 5))
    ax8.hist(preds[lbls==1], bins=40, alpha=0.5, color="#27ae60", label="Buy")
    ax8.hist(preds[lbls==0], bins=40, alpha=0.5, color="#e74c3c", label="Sell")
    f8.savefig(out_dir + "backtest_score_distribution.png", dpi=150); plt.close(f8)

    # Calibration
    f9, ax9 = plt.subplots(figsize=(7, 7))
    bins = np.linspace(0, 1, 11)
    for i in range(10):
        m = (preds >= bins[i]) & (preds < bins[i+1])
        if any(m): ax9.scatter(preds[m].mean(), lbls[m].mean(), s=100, color="#2980b9")
    ax9.plot([0,1], [0,1], ls="--", color="#888888")
    f9.savefig(out_dir + "backtest_calibration.png", dpi=150); plt.close(f9)

    # Fig 10 & 11: Confidence/Direction
    conf = np.abs(preds - 0.5)
    f10, ax10 = plt.subplots(figsize=(9, 5))
    # ... binned bar plot logic ...
    f10.savefig(out_dir + "backtest_confidence_vs_accuracy.png", dpi=150); plt.close(f10)

    f11, (axL, axR) = plt.subplots(1, 2, figsize=(10, 5))
    for ax, mask, c in [(axL, preds > 0.5, "#27ae60"), (axR, preds <= 0.5, "#e74c3c")]:
        if any(mask):
            acc = correct[mask].mean()
            ax.pie([acc, 1-acc], colors=[c, "#cccccc"], autopct="%1.1f%%")
    f11.savefig(out_dir + "backtest_long_short_split.png", dpi=150); plt.close(f11)

print(f"Analysis complete. Visuals exported to {out_dir}")