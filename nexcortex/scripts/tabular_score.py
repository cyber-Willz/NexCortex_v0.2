#!/usr/bin/env python3
"""Score tabular_eval outputs and baselines on identical held-out nodes. Usage: tabular_score.py GRAPH_DIR OUT_DIR N_SPLITS"""
import sys, numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
gd, od, ns = sys.argv[1], sys.argv[2], int(sys.argv[3])
nodes = pd.read_csv(f"{gd}/nodes.tsv", sep="\t", header=None, names=["id", "kind", "failed"])
sp = pd.read_csv(f"{gd}/splits.tsv", sep="\t", header=None)
edges = pd.read_csv(f"{gd}/edges.tsv", sep="\t", header=None, names=["a", "b", "w"])
X = np.load(f"{gd}/features.npy")
raw = pd.read_csv(f"{od}/accounting_data.csv")
ids = nodes["id"].tolist(); pos_of = {t: i for i, t in enumerate(ids)}
y = nodes["failed"].values
W = np.zeros((len(ids), len(ids)))
for a, b, w in edges.itertuples(index=False):
    W[a, b] = W[b, a] = w

def rank_metrics(order, y_test_by_id):
    """order: list of test node ids best-first."""
    lab = np.array([y_test_by_id[i] for i in order]); n = len(lab); P = lab.sum()
    cum = np.cumsum(lab)
    ap = float((cum[lab == 1] / (np.nonzero(lab)[0] + 1)).mean()) if P else 0.0
    ranks = np.nonzero(lab)[0] + 1                      # 1-based ranks of positives
    auc = float(1 - (ranks.mean() - (P + 1) / 2) / (n - P))
    k = lambda m: int(cum[min(m, n) - 1])
    return dict(auc=auc, ap=ap, p10=k(10) / 10, p30=k(30) / 30, rec30=k(30) / P, hit10=float(k(10) > 0))

rows = []        # (split, method, metrics)
clf_rows = []    # classification view at top-10% flagged
rng = np.random.default_rng(0)
def flag_metrics(order, y_by_id, frac=0.10):
    n = len(order); m = int(round(frac * n)); flagged = set(order[:m])
    t = np.array([y_by_id[i] for i in order]); f = np.array([i in flagged for i in order])
    tp = int((t & f).sum()); fp = int((~t.astype(bool) & f).sum()); fn = int((t & ~f).sum()); tn = n - tp - fp - fn
    P = t.sum()
    return dict(acc=(tp + tn) / n, prec=tp / max(tp + fp, 1), rec=tp / P, bal=(tp / P + tn / (n - P)) / 2, flagged=m, tp=tp)

meta = []
for s in range(ns):
    train = sp[1 + s].values == 1
    test_ids = [ids[i] for i in range(len(ids)) if not train[i]]
    ybid = {ids[i]: int(y[i]) for i in range(len(ids)) if not train[i]}
    meta.append((s, len(test_ids), sum(ybid.values())))
    # ---- baselines (score every test node once) ----
    lr = LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000).fit(X[train], y[train])
    gb = GradientBoostingClassifier(random_state=0, max_depth=2, n_estimators=100).fit(X[train], y[train])
    vol = raw["Transaction Volume"].values.astype(float)
    sign = 1 if vol[train & (y == 1)].mean() > vol[train & (y == 0)].mean() else -1
    ti = [pos_of[t] for t in test_ids]
    static = {"logreg (all 34 train failures)": lr.predict_proba(X[ti])[:, 1],
              "gradient boosting (all train failures)": gb.predict_proba(X[ti])[:, 1],
              "single feature: Transaction Volume": sign * vol[ti]}
    for name, sc in static.items():
        order = [test_ids[i] for i in np.argsort(-sc, kind="stable")]
        rows.append((s, name, rank_metrics(order, ybid))); clf_rows.append((s, name, flag_metrics(order, ybid)))
    rm = [rank_metrics(list(rng.permutation(test_ids)), ybid) for _ in range(500)]
    rows.append((s, "random (500 permutations)", {k: float(np.mean([r[k] for r in rm])) for k in rm[0]}))
    clf_rows.append((s, "random (500 permutations)", {k: float(np.mean([flag_metrics(list(rng.permutation(test_ids)), ybid)[k] for _ in range(200)])) for k in ["acc", "prec", "rec", "bal"]}))
    # ---- system outputs + per-incident guilt-by-association ----
    ev, rk = {}, {}
    for line in open(f"{od}/out{s}.tsv"):
        p = line.rstrip("\n").split("\t")
        if p[0] == "E": ev[int(p[2])] = p[3].split()
        elif p[0] == "R": rk.setdefault(p[3], {})[int(p[2])] = p[4].split()
    for inc, e in ev.items():
        sc = W[:, [pos_of[x] for x in e]].sum(1)[ti]
        rk.setdefault("tie weight to evidence (1-hop guilt by association)", {})[inc] = [test_ids[i] for i in np.argsort(-sc, kind="stable")]
    for name, d in rk.items():
        assert all(len(v) == len(test_ids) and set(v) == set(test_ids) for v in d.values()), name
        ms = [rank_metrics(v, ybid) for v in d.values()]; fm = [flag_metrics(v, ybid) for v in d.values()]
        rows.append((s, name, {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}))
        clf_rows.append((s, name, {k: float(np.mean([m[k] for m in fm])) for k in fm[0]}))

df = pd.DataFrame([dict(split=s, method=m, **v) for s, m, v in rows])
order = ["random (500 permutations)", "single feature: Transaction Volume", "logreg (all 34 train failures)", "gradient boosting (all train failures)",
         "tie weight to evidence (1-hop guilt by association)", "diffusion_rule", "reservoir_rule", "learned_all", "learned_graph", "learned_reservoir", "learned_human"]
print("test sets (split, nodes, failed):", meta)
print("\nRANKING METRICS: mean over splits (sd across splits). Base rate of failures in test = %.3f" % (np.mean([m[2] / m[1] for m in meta])))
agg = df.groupby("method").agg(["mean", "std"]).reindex(order)
cols = ["auc", "ap", "p10", "p30", "rec30", "hit10"]
out = pd.DataFrame({c: [f"{agg[(c,'mean')][m]:.3f} ({agg[(c,'std')][m]:.3f})" for m in order] for c in cols}, index=order)
print(out.to_string())
print("\nAUC per split:")
print(df.pivot(index="method", columns="split", values="auc").reindex(order).round(3).to_string())
cdf = pd.DataFrame([dict(split=s, method=m, **v) for s, m, v in clf_rows])
print("\nCLASSIFICATION VIEW: flag the top 10% of held-out transactions as 'failed' (mean over splits)")
ca = cdf.groupby("method")[["acc", "prec", "rec", "bal"]].mean().reindex(order)
print(ca.round(3).to_string())
tb = np.mean([1 - m[2] / m[1] for m in meta])
print(f"\ntrivial classifier 'everything succeeds': accuracy {tb:.3f}, recall of failures 0.000, balanced accuracy 0.500")
pd.concat([df]).to_csv(f"{od}/scores_by_split.csv", index=False)
# paired differences across splits
def paired(a, b, c):
    x = df[df.method == a].set_index("split")[c] - df[df.method == b].set_index("split")[c]
    return f"{x.mean():+.3f} (sd {x.std():.3f}; {int((x>0).sum())}/{len(x)} splits better)"
print("\nPAIRED (AUC): learned_all - reservoir_rule:", paired("learned_all", "reservoir_rule", "auc"))
print("PAIRED (AUC): learned_all - logreg:", paired("learned_all", "logreg (all 34 train failures)", "auc"))
print("PAIRED (AUC): reservoir_rule - random:", paired("reservoir_rule", "random (500 permutations)", "auc"))
print("PAIRED (AP): learned_all - reservoir_rule:", paired("learned_all", "reservoir_rule", "ap"))
print("PAIRED (AP): logreg - random:", paired("logreg (all 34 train failures)", "random (500 permutations)", "ap"))
