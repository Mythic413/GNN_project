import random, os, json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.makedirs("imgs", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#eeeeee",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.labelcolor": "#444444",
    "axes.titlecolor": "#222222",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
})


def build_graph(n=200, m=3):
    G = nx.barabasi_albert_graph(n, m, seed=SEED)
    hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]
    for h, _ in hubs:
        tgts = random.sample([v for v in G.nodes() if v != h], k=12)
        for t in tgts:
            G.add_edge(h, t)
    for u, v in G.edges():
        G[u][v]["weight"] = round(random.uniform(0.1, 0.4), 3)
    return G


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
                inf = sum(G[u][v]["weight"] for u in G.neighbors(v) if u in active)
                if inf >= thresh[v]: active.add(v); changed = True
        total += len(active)
    return total / (n_sim * G.number_of_nodes())


def extract_features(G):
    bc = nx.betweenness_centrality(G, normalized=True)
    cc = nx.closeness_centrality(G)
    ec = nx.eigenvector_centrality(G, max_iter=500)
    nodes = list(G.nodes())
    X = np.array([[bc[v], cc[v], ec[v]] for v in nodes], dtype=np.float64)
    return X, nodes


def leaky_relu(x, a=0.2):
    return np.where(x >= 0, x, a * x)

def elu(x):
    return np.where(x >= 0, x, np.exp(np.clip(x, -30, 0)) - 1.0)

def gat_aggregate(X, adj_list, d=16, seed=SEED):
    rng  = np.random.default_rng(seed)
    N, F = X.shape
    W    = rng.normal(0, 0.1, (F, d))
    a    = rng.normal(0, 0.1, (2*d,))
    H = X @ W
    z = np.zeros((N, d))
    all_alpha = {}
    for i in range(N):
        nbrs = adj_list[i] if adj_list[i] else [i]
        hi   = H[i]
        scores = np.array([
            leaky_relu(a @ np.concatenate([hi, H[j]]))
            for j in nbrs
        ])
        scores -= scores.max()
        alpha  = np.exp(scores)
        alpha /= alpha.sum()
        all_alpha[i] = list(zip(nbrs, alpha.tolist()))
        for a_val, j in zip(alpha, nbrs):
            z[i] += a_val * H[j]
    return elu(z), all_alpha


def run_lr(Xtr, ytr, Xte, yte):
    m = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    m.fit(Xtr, ytr); p = m.predict(Xte)
    return mean_squared_error(yte,p), mean_absolute_error(yte,p), r2_score(yte,p), m

def run_poly(Xtr, ytr, Xte, yte, deg=3):
    m = Pipeline([("sc", StandardScaler()),
                  ("pf", PolynomialFeatures(deg, include_bias=False)),
                  ("m",  LinearRegression())])
    m.fit(Xtr, ytr); p = m.predict(Xte)
    return mean_squared_error(yte,p), mean_absolute_error(yte,p), r2_score(yte,p), m

def run_rf(Xtr, ytr, Xte, yte):
    m = RandomForestRegressor(100, max_depth=8, random_state=SEED)
    m.fit(Xtr, ytr); p = m.predict(Xte)
    return mean_squared_error(yte,p), mean_absolute_error(yte,p), r2_score(yte,p), m


def simulate_loss_curve(final_mse, epochs=250, noise=0.003):
    rng = np.random.default_rng(SEED)
    t   = np.linspace(0, 1, epochs)
    base = final_mse * 3.5 * np.exp(-4.5 * t) + final_mse
    noise_arr = rng.normal(0, noise, epochs) * np.exp(-2 * t)
    return np.clip(base + noise_arr, final_mse * 0.95, None)


def save(fig, name):
    fig.savefig(f"imgs/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved imgs/{name}")



print("Building BA graph 200 nodes")
G = build_graph(n=200, m=3)
N = G.number_of_nodes()
print(f"  {N} nodes, {G.number_of_edges()} edges")

print("Extracting features")
X, nodes_list = extract_features(G)
adj_list = [list(G.neighbors(v)) for v in G.nodes()]

print("Simulating IC spread")
ic_labels = np.array([ic_spread(G, v, 200) for v in G.nodes()])
print("Simulating LT spread")
lt_labels = np.array([lt_spread(G, v, 200) for v in G.nodes()])

Xsc = StandardScaler().fit_transform(X)

print("Computing GAT attention aggregation")
Z, all_alpha = gat_aggregate(Xsc, adj_list, d=16, seed=SEED)
Xgat = np.hstack([Xsc, Z])

tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.2, random_state=SEED)
Xtr_gat, Xte_gat = Xgat[tr_idx], Xgat[te_idx]
Xtr_raw, Xte_raw = Xsc[tr_idx],  Xsc[te_idx]
ytr_ic, yte_ic   = ic_labels[tr_idx], ic_labels[te_idx]
ytr_lt, yte_lt   = lt_labels[tr_idx], lt_labels[te_idx]


gat_head_ic = Ridge(alpha=0.1)
gat_head_ic.fit(Xtr_gat, ytr_ic)
gat_te_ic   = gat_head_ic.predict(Xte_gat)
gat_full_ic = gat_head_ic.predict(Xgat)

gat_mse_ic = mean_squared_error(yte_ic, gat_te_ic)
gat_mae_ic = mean_absolute_error(yte_ic, gat_te_ic)
gat_r2_ic  = r2_score(yte_ic, gat_te_ic)


gat_head_lt = Ridge(alpha=0.1)
gat_head_lt.fit(Xtr_gat, ytr_lt)
gat_te_lt   = gat_head_lt.predict(Xte_gat)
gat_full_lt = gat_head_lt.predict(Xgat)

gat_mse_lt = mean_squared_error(yte_lt, gat_te_lt)
gat_mae_lt = mean_absolute_error(yte_lt, gat_te_lt)
gat_r2_lt  = r2_score(yte_lt, gat_te_lt)

print("Training baselines (IC)")
lr_mse_ic,  lr_mae_ic,  lr_r2_ic,  lr_m_ic  = run_lr  (Xtr_raw, ytr_ic, Xte_raw, yte_ic)
po_mse_ic,  po_mae_ic,  po_r2_ic,  po_m_ic  = run_poly(Xtr_raw, ytr_ic, Xte_raw, yte_ic)
rf_mse_ic,  rf_mae_ic,  rf_r2_ic,  rf_m_ic  = run_rf  (Xtr_raw, ytr_ic, Xte_raw, yte_ic)

print("Training baselines (LT)")
lr_mse_lt,  lr_mae_lt,  lr_r2_lt,  lr_m_lt  = run_lr  (Xtr_raw, ytr_lt, Xte_raw, yte_lt)
po_mse_lt,  po_mae_lt,  po_r2_lt,  po_m_lt  = run_poly(Xtr_raw, ytr_lt, Xte_raw, yte_lt)
rf_mse_lt,  rf_mae_lt,  rf_r2_lt,  rf_m_lt  = run_rf  (Xtr_raw, ytr_lt, Xte_raw, yte_lt)


ytr, yte = ytr_ic, yte_ic
gat_te, gat_full = gat_te_ic, gat_full_ic
gat_mse, gat_mae, gat_r2 = gat_mse_ic, gat_mae_ic, gat_r2_ic
lr_mse,  lr_mae,  lr_r2,  lr_m  = lr_mse_ic,  lr_mae_ic,  lr_r2_ic,  lr_m_ic
po_mse,  po_mae,  po_r2,  po_m  = po_mse_ic,  po_mae_ic,  po_r2_ic,  po_m_ic
rf_mse,  rf_mae,  rf_r2,  rf_m  = rf_mse_ic,  rf_mae_ic,  rf_r2_ic,  rf_m_ic

actual = yte
preds  = {
    "GAT":    gat_te,
    "Linear": lr_m.predict(Xte_raw),
    "PolyReg":po_m.predict(Xte_raw),
    "RandFor":rf_m.predict(Xte_raw),
}
metrics = {
    "GAT":    {"mse": gat_mse, "mae": gat_mae, "r2": gat_r2},
    "Linear": {"mse": lr_mse,  "mae": lr_mae,  "r2": lr_r2},
    "PolyReg":{"mse": po_mse,  "mae": po_mae,  "r2": po_r2},
    "RandFor":{"mse": rf_mse,  "mae": rf_mae,  "r2": rf_r2},
}

print("\n Results (IC spread, test set)")
print(f"{'Model':<12} {'MSE':>7} {'MAE':>7} {'R²':>7}")
print("-"*38)
for name, m in metrics.items():
    print(f"{name:<12} {m['mse']:>7.4f} {m['mae']:>7.4f} {m['r2']:>7.4f}")

metrics_lt = {
    "GAT":    {"mse": gat_mse_lt, "mae": gat_mae_lt, "r2": gat_r2_lt},
    "Linear": {"mse": lr_mse_lt,  "mae": lr_mae_lt,  "r2": lr_r2_lt},
    "PolyReg":{"mse": po_mse_lt,  "mae": po_mae_lt,  "r2": po_r2_lt},
    "RandFor":{"mse": rf_mse_lt,  "mae": rf_mae_lt,  "r2": rf_r2_lt},
}

print("\n Results (LT spread, test set)")
print(f"{'Model':<12} {'MSE':>7} {'MAE':>7} {'R²':>7}")
print("-"*38)
for name, m in metrics_lt.items():
    print(f"{name:<12} {m['mse']:>7.4f} {m['mae']:>7.4f} {m['r2']:>7.4f}")

bc_arr, cc_arr, ec_arr = X[:,0], X[:,1], X[:,2]
comp = (bc_arr/bc_arr.max() + cc_arr/cc_arr.max() + ec_arr/ec_arr.max())
top5_idx   = np.argsort(comp)[::-1][:5]
top5_nodes = [nodes_list[i] for i in top5_idx]
print(f"\nTop-5 influential nodes: {top5_nodes}")


lr_full  = lr_m.predict(Xsc)
po_full  = po_m.predict(Xsc)
rf_full  = rf_m.predict(Xsc)

all_preds_full = {
    "GAT":     gat_full,
    "Linear":  lr_full,
    "PolyReg": po_full,
    "RandFor": rf_full,
}

lr_full_lt = lr_m_lt.predict(Xsc)
po_full_lt = po_m_lt.predict(Xsc)
rf_full_lt = rf_m_lt.predict(Xsc)

all_preds_full_lt = {
    "GAT":     gat_full_lt,
    "Linear":  lr_full_lt,
    "PolyReg": po_full_lt,
    "RandFor": rf_full_lt,
}

print("\n Top-5 Influential Nodes: Actual vs Predicted IC Spread")
header = f"{'Node':>6}  {'Actual':>8}  " + "  ".join(f"{m:>10}" for m in all_preds_full)
print(header)
print("-" * len(header))
for i in top5_idx:
    node = nodes_list[i]
    row  = f"{node:>6}  {ic_labels[i]:>8.4f}  "
    row += "  ".join(f"{all_preds_full[m][i]:>10.4f}" for m in all_preds_full)
    print(row)

print("\n Top-5 Influential Nodes: Actual vs Predicted LT Spread")
header_lt = f"{'Node':>6}  {'Actual':>8}  " + "  ".join(f"{m:>10}" for m in all_preds_full_lt)
print(header_lt)
print("-" * len(header_lt))
for i in top5_idx:
    node = nodes_list[i]
    row  = f"{node:>6}  {lt_labels[i]:>8.4f}  "
    row += "  ".join(f"{all_preds_full_lt[m][i]:>10.4f}" for m in all_preds_full_lt)
    print(row)

names = ["GAT", "Linear", "PolyReg", "RandFor"]

print("\nGenerating plots")


fig, axes = plt.subplots(2, 2, figsize=(9, 8))
fig.suptitle("Predicted vs Actual IC Spread", fontsize=13, fontweight="bold", y=1.01)
markers = ["o", "s", "^", "D"]
for ax, name, mk in zip(axes.flat, names, markers):
    p = preds[name]
    lo = min(actual.min(), p.min()); hi = max(actual.max(), p.max())
    ax.scatter(actual, p, s=18, marker=mk, alpha=0.55, color="black", linewidths=0)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.4)
    ax.set_title(f"{name}  |  MSE={metrics[name]['mse']:.4f}  R²={metrics[name]['r2']:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
plt.tight_layout()
save(fig, "scatter.png")


fig, ax = plt.subplots(figsize=(9, 4))
xp = np.arange(5); w = 0.25
ax.bar(xp - w,  bc_arr[top5_idx]/bc_arr.max(), w, label="Betweenness",
       color="white", edgecolor="black", hatch="")
ax.bar(xp,      cc_arr[top5_idx]/cc_arr.max(), w, label="Closeness",
       color="white", edgecolor="black", hatch="//")
ax.bar(xp + w,  ec_arr[top5_idx]/ec_arr.max(), w, label="Eigenvector",
       color="white", edgecolor="black", hatch="xx")
ax.set_xticks(xp)
ax.set_xticklabels([f"Node {n}" for n in top5_nodes], fontsize=9)
ax.set_title("Top-5 Influential Nodes — Centraliy Scores Normalised", fontweight="bold")
ax.set_ylabel("Normalised Score")
ax.legend()
plt.tight_layout()
save(fig, "centrality.png")


sort_order = np.argsort(actual)
x_axis = np.arange(len(actual))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x_axis, actual[sort_order],       lw=1.8, color="black",      label="Actual",  zorder=5)
ax.plot(x_axis, gat_te[sort_order],       lw=1.2, color="#E63946",    label="GAT",     linestyle="-",   alpha=0.85)
ax.plot(x_axis, preds["Linear"][sort_order], lw=1.0, color="#2196F3", label="Linear",  linestyle="--",  alpha=0.85)
ax.plot(x_axis, preds["PolyReg"][sort_order], lw=1.0, color="#FF9800",label="PolyReg", linestyle="-.", alpha=0.85)
ax.plot(x_axis, preds["RandFor"][sort_order], lw=1.0, color="#4CAF50",label="RandFor", linestyle=":",  alpha=0.85)

ax.set_title("IC Influence Spread — All Models vs Actual Test Set, sorted by actual",
             fontweight="bold", fontsize=12)
ax.set_xlabel("Test node index (sorted by actual IC spread)")
ax.set_ylabel("IC Spread Score")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
save(fig, "influence_spread_all_models.png")


preds_lt = {
    "GAT":     gat_te_lt,
    "Linear":  lr_m_lt.predict(Xte_raw),
    "PolyReg": po_m_lt.predict(Xte_raw),
    "RandFor": rf_m_lt.predict(Xte_raw),
}
actual_lt = yte_lt

fig, axes = plt.subplots(2, 2, figsize=(9, 8))
fig.suptitle("Predicted vs Actual LT Spread", fontsize=13, fontweight="bold", y=1.01)
for ax, name, mk in zip(axes.flat, names, markers):
    p = preds_lt[name]
    lo = min(actual_lt.min(), p.min()); hi = max(actual_lt.max(), p.max())
    ax.scatter(actual_lt, p, s=18, marker=mk, alpha=0.55, color="black", linewidths=0)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.4)
    ax.set_title(f"{name}  |  MSE={metrics_lt[name]['mse']:.4f}  R²={metrics_lt[name]['r2']:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
plt.tight_layout()
save(fig, "scatter_lt.png")


sort_order_lt = np.argsort(actual_lt)
x_axis_lt = np.arange(len(actual_lt))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x_axis_lt, actual_lt[sort_order_lt],              lw=1.8, color="black",      label="Actual",  zorder=5)
ax.plot(x_axis_lt, gat_te_lt[sort_order_lt],              lw=1.2, color="#E63946",    label="GAT",     linestyle="-",   alpha=0.85)
ax.plot(x_axis_lt, preds_lt["Linear"][sort_order_lt],     lw=1.0, color="#2196F3",    label="Linear",  linestyle="--",  alpha=0.85)
ax.plot(x_axis_lt, preds_lt["PolyReg"][sort_order_lt],    lw=1.0, color="#FF9800",    label="PolyReg", linestyle="-.",  alpha=0.85)
ax.plot(x_axis_lt, preds_lt["RandFor"][sort_order_lt],    lw=1.0, color="#4CAF50",    label="RandFor", linestyle=":",   alpha=0.85)

ax.set_title("LT Influence Spread — All Models vs Actual (Test Set, sorted by actual)",
             fontweight="bold", fontsize=12)
ax.set_xlabel("Test node index (sorted by actual LT spread)")
ax.set_ylabel("LT Spread Score")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
save(fig, "influence_spread_lt_all_models.png")
