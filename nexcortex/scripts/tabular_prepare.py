#!/usr/bin/env python3
"""Turn accounting_data.csv into a transaction-similarity graph + repeated stratified splits.

nodes.tsv  : node id <TAB> kind (account type) <TAB> failed (1 if Transaction Outcome == 0)
edges.tsv  : i <TAB> j <TAB> weight           (symmetric k-nearest-neighbour ties in standardised feature space; NO labels used)
splits.tsv : node id <TAB> train flag for each of N repeated 70/30 stratified splits
Usage: tabular_prepare.py accounting_data.csv OUT_DIR [k=10] [n_splits=5]
"""
import sys, os, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit
src, out = sys.argv[1], sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
ns = int(sys.argv[4]) if len(sys.argv) > 4 else 5
os.makedirs(out, exist_ok=True)
df = pd.read_csv(src)
failed = (df["Transaction Outcome"] == 0).astype(int).values
X = df.drop(columns=["Transaction ID", "Date", "Transaction Outcome", "Normalized Transaction Amount"])
X["Missing Data Indicator"] = X["Missing Data Indicator"].astype(int)
X = pd.get_dummies(X, columns=["Account Type"]).astype(float).values
X = (X - X.mean(0)) / X.std(0)
nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
d, ix = nn.kneighbors(X)
med = np.median(d[:, 1:] ** 2)
edges = {}
for i in range(len(X)):
    for dist, j in zip(d[i, 1:], ix[i, 1:]):
        a, b = (i, j) if i < j else (j, i)
        edges[(a, b)] = max(edges.get((a, b), 0.0), float(np.exp(-dist ** 2 / med)))
ids = [f"t{int(t):04d}" for t in df["Transaction ID"]]
with open(f"{out}/nodes.tsv", "w") as f:
    for i in range(len(ids)):
        f.write(f"{ids[i]}\t{df['Account Type'][i]}\t{failed[i]}\n")
with open(f"{out}/edges.tsv", "w") as f:
    for (a, b), w in sorted(edges.items()):
        f.write(f"{a}\t{b}\t{w:.6f}\n")
sp = np.zeros((len(ids), ns), dtype=int)
for s in range(ns):
    tr, _ = next(StratifiedShuffleSplit(1, test_size=0.3, random_state=s).split(X, failed))
    sp[tr, s] = 1
with open(f"{out}/splits.tsv", "w") as f:
    for i in range(len(ids)):
        f.write(ids[i] + "\t" + "\t".join(map(str, sp[i])) + "\n")
np.save(f"{out}/features.npy", X)
print(f"{len(ids)} nodes, {len(edges)} ties (k={k}), {failed.sum()} failed, {ns} splits (train failed per split: {[int((sp[:,s]*failed).sum()) for s in range(ns)]})")
