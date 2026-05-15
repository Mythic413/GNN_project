import random, os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.makedirs("imgs", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc", "axes.grid": True,
    "grid.color": "#eeeeee", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.labelcolor": "#444444", "axes.titlecolor": "#222222",
    "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
})

COLORS = {
    "GAT": "#E63946", "GCN": "#2196F3", "GIN": "#FF9800", "GraphSAGE": "#9C27B0",
    "Linear": "#4CAF50", "PolyReg": "#795548", "RandFor": "#607D8B",
}

def leaky_relu(x, a=0.2): return np.where(x >= 0, x, a * x)
def elu(x): return np.where(x >= 0, x, np.exp(np.clip(x, -30, 0)) - 1.0)
def relu(x): return np.maximum(0, x)

def build_graph(n=500, m=3):
    G = nx.barabasi_albert_graph(n, m, seed=SEED)
    for h, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]:
        for t in random.sample([v for v in G.nodes() if v != h], k=12):
            G.add_edge(h, t)
    for u, v in G.edges():
        G[u][v]["weight"] = round(random.uniform(0.1, 0.4), 3)
    return G

def extract_features(G):
    bc = nx.betweenness_centrality(G, normalized=True)
    cc = nx.closeness_centrality(G)
    ec = nx.eigenvector_centrality(G, max_iter=500)
    nodes = list(G.nodes())
    return np.array([[bc[v], cc[v], ec[v]] for v in nodes], dtype=np.float64), nodes

def ic_spread(G, seed_node, n_sim=200):
    total = 0
    for _ in range(n_sim):
        active = {seed_node}; frontier = deque([seed_node])
        while frontier:
            u = frontier.popleft()
            for v in G.neighbors(u):
                if v not in active and random.random() < G[u][v]["weight"]:
                    active.add(v); frontier.append(v)
        total += len(active)
    return total / (n_sim * G.number_of_nodes())

def lt_spread(G, seed_node, n_sim=200):
    total = 0
    for _ in range(n_sim):
        thresh = {v: random.random() for v in G.nodes()}
        active, changed = {seed_node}, True
        while changed:
            changed = False
            for v in G.nodes():
                if v in active: continue
                if sum(G[u][v]["weight"] for u in G.neighbors(v) if u in active) >= thresh[v]:
                    active.add(v); changed = True
        total += len(active)
    return total / (n_sim * G.number_of_nodes())

def gat_aggregate(X, adj_list, d=16, seed=SEED):
    rng = np.random.default_rng(seed)
    N, F = X.shape
    W = rng.normal(0, 0.1, (F, d)); a = rng.normal(0, 0.1, (2*d,)); H = X @ W
    Z = np.zeros((N, d))
    for i in range(N):
        nbrs = adj_list[i] if adj_list[i] else [i]
        scores = np.array([leaky_relu(a @ np.concatenate([H[i], H[j]])) for j in nbrs])
        scores -= scores.max(); alpha = np.exp(scores); alpha /= alpha.sum()
        for av, j in zip(alpha, nbrs): Z[i] += av * H[j]
    return elu(Z)

def gcn_aggregate(X, adj_list, d=16, seed=SEED):
    rng = np.random.default_rng(seed)
    N, F = X.shape; W = rng.normal(0, 0.1, (F, d))
    A = np.zeros((N, N))
    for i, nbrs in enumerate(adj_list):
        A[i, i] = 1.0
        for j in nbrs: A[i, j] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-8)))
    return relu((D_inv_sqrt @ A @ D_inv_sqrt) @ X @ W)

def gin_aggregate(X, adj_list, d=16, seed=SEED):
    rng = np.random.default_rng(seed)
    N, F = X.shape
    W1 = rng.normal(0, 0.1, (F, d)); W2 = rng.normal(0, 0.1, (d, d))
    agg = np.zeros((N, F))
    for i in range(N):
        if adj_list[i]: agg[i] = X[adj_list[i]].sum(axis=0)
    H = relu(((1.0) * X + agg) @ W1); return relu(H @ W2)

def graphsage_aggregate(X, adj_list, d=16, seed=SEED):
    rng = np.random.default_rng(seed)
    N, F = X.shape; W = rng.normal(0, 0.1, (2*F, d))
    agg = np.array([X[adj_list[i]].mean(axis=0) if adj_list[i] else X[i] for i in range(N)])
    H = relu(np.hstack([X, agg]) @ W)
    return H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-8)

def run_gnn(agg_fn, X, y, tr_idx, te_idx, adj_list, d=16):
    scaler = StandardScaler().fit(X[tr_idx]); Xs = scaler.transform(X)
    feat = np.hstack([agg_fn(Xs, adj_list, d=d), Xs])
    m = Ridge(alpha=0.1).fit(feat[tr_idx], y[tr_idx])
    return m, feat

def run_baseline(pipe, Xtr, ytr, Xte): pipe.fit(Xtr, ytr); return pipe

G = build_graph()
X, nodes_list = extract_features(G)
adj_list = [list(G.neighbors(v)) for v in nodes_list]
ic_labels = np.array([ic_spread(G, v) for v in nodes_list])
lt_labels = np.array([lt_spread(G, v) for v in nodes_list])

idx = np.arange(len(nodes_list))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)
Xtr_raw, Xte_raw = X[train_idx], X[test_idx]

scaler = StandardScaler().fit(X[train_idx]); X_scaled = scaler.transform(X)

model_names = ["GAT", "GCN", "GIN", "GraphSAGE", "Linear", "PolyReg", "RandFor"]

def get_preds(labels):
    ytr, yte = labels[train_idx], labels[test_idx]
    preds = {}
    for name, fn in [("GAT", gat_aggregate), ("GCN", gcn_aggregate),
                     ("GIN", gin_aggregate), ("GraphSAGE", graphsage_aggregate)]:
        m, feat = run_gnn(fn, X, labels, train_idx, test_idx, adj_list)
        preds[name] = m.predict(feat[test_idx])
    preds["Linear"]  = run_baseline(Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]), Xtr_raw, ytr, Xte_raw).predict(Xte_raw)
    preds["PolyReg"] = run_baseline(Pipeline([("sc", StandardScaler()), ("poly", PolynomialFeatures(3)), ("m", LinearRegression())]), Xtr_raw, ytr, Xte_raw).predict(Xte_raw)
    preds["RandFor"] = run_baseline(RandomForestRegressor(100, max_depth=8, random_state=SEED), Xtr_raw, ytr, Xte_raw).predict(Xte_raw)
    return preds, yte

preds_ic, yte_ic = get_preds(ic_labels)
preds_lt, yte_lt = get_preds(lt_labels)

bc, cc, ec = X[:, 0], X[:, 1], X[:, 2]
top5_idx = np.argsort(bc/bc.max() + cc/cc.max() + ec/ec.max())[::-1][:5]
top5_nodes = [nodes_list[i] for i in top5_idx]

def save(fig, name):
    fig.savefig(f"imgs/{name}", dpi=150, bbox_inches="tight"); plt.close(fig)

def line_plot(preds, yte, title, ylabel, fname):
    s = np.argsort(yte); x = np.arange(len(yte))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, yte[s], lw=2.2, color="black", label="Actual", zorder=10)
    for nm in model_names:
        ls = "-" if nm in ["GAT", "GCN", "GIN", "GraphSAGE"] else "--"
        ax.plot(x, preds[nm][s], lw=1.2, color=COLORS[nm], label=nm, linestyle=ls, alpha=0.85)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("Test node index (sorted by actual)"); ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    plt.tight_layout(); save(fig, fname)

line_plot(preds_ic, yte_ic, "IC Influence Spread — All Models vs Actual (sorted by actual)", "IC Spread Score", "influence_spread_ic.png")
line_plot(preds_lt, yte_lt, "LT Influence Spread — All Models vs Actual (sorted by actual)", "LT Spread Score", "influence_spread_lt.png")

fig, ax = plt.subplots(figsize=(9, 4))
xp = np.arange(5); w = 0.25
ax.bar(xp - w, bc[top5_idx]/bc.max(), w, label="Betweenness", color="white", edgecolor="black", hatch="")
ax.bar(xp,     cc[top5_idx]/cc.max(), w, label="Closeness",   color="white", edgecolor="black", hatch="//")
ax.bar(xp + w, ec[top5_idx]/ec.max(), w, label="Eigenvector", color="white", edgecolor="black", hatch="xx")
ax.set_xticks(xp); ax.set_xticklabels([f"Node {n}" for n in top5_nodes], fontsize=9)
ax.set_title("Top-5 Influential Nodes — Centrality Scores (Normalised)", fontweight="bold")
ax.set_ylabel("Normalised Score"); ax.legend()
plt.tight_layout(); save(fig, "centrality.png")

import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def metrics(preds, yte):
    return {nm: {"mse": float(mean_squared_error(yte, p)), "mae": float(mean_absolute_error(yte, p)), "r2": float(r2_score(yte, p))} for nm, p in preds.items()}

results = {
    "model_names": model_names,
    "metrics_ic": metrics(preds_ic, yte_ic),
    "metrics_lt": metrics(preds_lt, yte_lt),
    "top5_nodes": [int(n) for n in top5_nodes],
    "top5_idx":   [int(i) for i in top5_idx],
    "ic_labels_top5": [float(ic_labels[i]) for i in top5_idx],
    "lt_labels_top5": [float(lt_labels[i]) for i in top5_idx],
}
with open("imgs/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved: imgs/influence_spread_ic.png, imgs/influence_spread_lt.png, imgs/centrality.png, imgs/results.json")