#!/usr/bin/env python3
"""ai_accounting_dataset.csv -> bipartite-style graph: transactions linked to attribute entities.
Entities: one per transaction (kind 'transaction', label = is_fraud) + one per vendor / category / account /
payment method / description / amount band (5 quantiles) / month. Ties transaction--attribute, weight 1. No labels used.
Outputs nodes.tsv, edges.tsv, splits.tsv (stratified 70/30 over TRANSACTIONS; attribute nodes always 'train', never scored).
Usage: fraud_prepare.py data.csv OUT_DIR [n_splits=15]"""
import sys, os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
src, out = sys.argv[1], sys.argv[2]; ns = int(sys.argv[3]) if len(sys.argv) > 3 else 15
os.makedirs(out, exist_ok=True)
df = pd.read_csv(src)
df["band"] = pd.qcut(df.amount, 5, labels=False, duplicates="drop").astype(int)
df["month"] = pd.to_datetime(df.date).dt.month
safe = lambda s: str(s).replace(" ", "_").replace("'", "")
attrs = {"vendor": "vendor", "category": "category", "account": "account", "payment_method": "payment_method", "description": "description", "band": "amount_band", "month": "month"}
nodes = [(r.transaction_id, "transaction", int(r.is_fraud)) for r in df.itertuples()]
idx = {n[0]: i for i, n in enumerate(nodes)}
edges = []
for col, kind in attrs.items():
    for v in sorted(df[col].unique(), key=str):
        name = f"{kind}:{safe(v)}"; idx[name] = len(nodes); nodes.append((name, kind, 0))
    for r in df.itertuples():
        edges.append((idx[r.transaction_id], idx[f"{kind}:{safe(getattr(r, col))}"], 1.0))
n_tx = len(df); y = df.is_fraud.values
sp = np.ones((len(nodes), ns), dtype=int)
for s in range(ns):
    tr, te = next(StratifiedShuffleSplit(1, test_size=0.3, random_state=s).split(np.zeros(n_tx), y))
    sp[te, s] = 0
with open(f"{out}/nodes.tsv", "w") as f:
    for n in nodes: f.write(f"{n[0]}\t{n[1]}\t{n[2]}\n")
with open(f"{out}/edges.tsv", "w") as f:
    for a, b, w in edges: f.write(f"{a}\t{b}\t{w}\n")
with open(f"{out}/splits.tsv", "w") as f:
    for i, n in enumerate(nodes): f.write(n[0] + "\t" + "\t".join(map(str, sp[i])) + "\n")
print(f"{len(nodes)} entities ({n_tx} transactions + {len(nodes)-n_tx} attributes), {len(edges)} ties, {y.sum()} fraud, {ns} splits; test frauds per split: {sorted(set(int(y[sp[:n_tx,s]==0].sum()) for s in range(ns)))}")
