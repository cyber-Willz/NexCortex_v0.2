#!/usr/bin/env python3
"""Score tabular_eval outputs for the fraud dataset + baselines. Usage: fraud_score.py GRAPH_DIR OUT_DIR CSV N_SPLITS"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
gd, od, csv, ns = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sims = int(sys.argv[5]) if len(sys.argv) > 5 else 5000
quiet = len(sys.argv) > 6
nodes = pd.read_csv(f"{gd}/nodes.tsv", sep="\t", header=None, names=["id", "kind", "f"])
sp = pd.read_csv(f"{gd}/splits.tsv", sep="\t", header=None)
edges = pd.read_csv(f"{gd}/edges.tsv", sep="\t", header=None, names=["a", "b", "w"])
raw = pd.read_csv(csv)
ids = nodes.id.tolist(); pos_of = {t: i for i, t in enumerate(ids)}
ntx = int((nodes.kind == "transaction").sum()); y = nodes.f.values[:ntx]
attrs = {i: set() for i in range(ntx)}
for a, b, w in edges.itertuples(index=False): attrs[a].add(b)
cats = ["vendor", "category", "account", "payment_method"]
Xcat = pd.get_dummies(raw[cats].assign(band=pd.qcut(raw.amount, 5, labels=False), month=pd.to_datetime(raw.date).dt.month).astype(str)).astype(float)
Xcat["logamt"] = np.log(raw.amount); Xcat["logamt"] = (Xcat.logamt - Xcat.logamt.mean()) / Xcat.logamt.std()
Xc = Xcat.values

def from_ranks(r, n):      # r: 1-based ranks of positives
    r = np.sort(np.asarray(r)); P = len(r)
    auc = 1 - (r.mean() - (P + 1) / 2) / (n - P)
    ap = np.mean([(j + 1) / rj for j, rj in enumerate(r)])
    return dict(auc=auc, ap=ap, hit10=float((r <= 10).any()), hit15=float((r <= 15).any()), pct=float(r.mean() / n), top10pct_rec=float((r <= round(0.1 * n)).sum() / P))
def flagm(r, n, frac=0.1):
    m = round(frac * n); P = len(r); tp = int((np.asarray(r) <= m).sum()); fp = m - tp; tn = n - P - fp
    return dict(acc=(tp + tn) / n, prec=tp / m, rec=tp / P, bal=(tp / P + tn / (n - P)) / 2)
def ranks_of(order, ybid): return [i + 1 for i, t in enumerate(order) if ybid[t]]

rng = np.random.default_rng(0)
M, C, per_fraud = [], [], {}
for s in range(ns):
    tr = sp[1 + s].values[:ntx] == 1; te = np.where(~tr)[0]; tr_i = np.where(tr)[0]
    tid = [ids[i] for i in te]; ybid = {ids[i]: int(y[i]) for i in te}; n = len(te)
    base = {}
    lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000).fit(Xc[tr_i], y[tr_i]); base["logistic regression (one-hot + amount)"] = lr.predict_proba(Xc[te])[:, 1]
    # smoothed target encoding / naive Bayes on categorical attributes, train rows only
    pri = y[tr_i].mean(); lo = lambda p: np.log(p / (1 - p))
    nb = np.zeros(n); vend = np.zeros(n)
    for c in cats:
        g = raw.iloc[tr_i].groupby(c).is_fraud.agg(["sum", "count"]); rate = ((g["sum"] + 5 * pri) / (g["count"] + 5)).to_dict()
        sc = np.array([lo(rate.get(v, pri)) - lo(pri) for v in raw[c].values[te]]); nb += sc
        if c == "vendor": vend = sc
    base["vendor fraud-rate prior (train)"] = vend; base["naive Bayes on vendor/category/account/payment"] = nb
    for name, sc in base.items():
        o = [tid[i] for i in np.argsort(-sc, kind="stable")]; r = ranks_of(o, ybid); M.append((s, name, from_ranks(r, n))); C.append((s, name, flagm(r, n)))
        if name.startswith("logistic"): pass
    # system rankings
    ev, rk = {}, {}
    for line in open(f"{od}/out{s}.tsv"):
        p = line.rstrip("\n").split("\t")
        if p[0] == "E": ev[int(p[2])] = p[3].split()
        elif p[0] == "R": rk.setdefault(p[3], {})[int(p[2])] = p[4].split()
    for inc, e in ev.items():
        es = set().union(*[attrs[pos_of[x]] for x in e]) if False else None
        cnt = np.array([sum(len(attrs[i] & attrs[pos_of[x]]) for x in e) for i in te])
        rk.setdefault("shared attributes with the evidence frauds", {})[inc] = [tid[i] for i in np.argsort(-cnt, kind="stable")]
    for name, d in rk.items():
        assert all(set(v) == set(tid) for v in d.values()), name
        rs = [from_ranks(ranks_of(v, ybid), n) for v in d.values()]; fs = [flagm(ranks_of(v, ybid), n) for v in d.values()]
        M.append((s, name, {k: float(np.mean([r[k] for r in rs])) for k in rs[0]})); C.append((s, name, {k: float(np.mean([f[k] for f in fs])) for k in fs[0]}))
        if name in ("reservoir_rule", "learned_all"):
            for v in d.values():
                for t in tid:
                    if ybid[t]: per_fraud.setdefault((name, t), []).append((v.index(t) + 1) / n)
df = pd.DataFrame([dict(split=s, method=m, **v) for s, m, v in M]); cdf = pd.DataFrame([dict(split=s, method=m, **v) for s, m, v in C])
# null: random rankings, same test sizes / positives (2 frauds in 150)
n_te = len(te); P = int(sum(ybid.values()))
null = []
for _ in range(sims):
    vals = [from_ranks(rng.choice(n_te, P, replace=False) + 1, n_te) for _ in range(ns)]
    null.append({k: np.mean([v[k] for v in vals]) for k in vals[0]})
null = pd.DataFrame(null)
order = ["logistic regression (one-hot + amount)", "naive Bayes on vendor/category/account/payment", "vendor fraud-rate prior (train)", "shared attributes with the evidence frauds",
         "diffusion_rule", "reservoir_rule", "learned_all", "learned_graph", "learned_reservoir", "learned_human"]
if quiet:
    for m in order: print("MEAN", m, round(df[df.method == m].auc.mean(), 4), round(df[df.method == m].ap.mean(), 4))
    sys.exit(0)
print(f"splits {ns}; test set {n_te} transactions with {P} fraud each; frauds overall {int(y.sum())}; base rate {P/n_te:.4f}")
print("\nRANDOM (null, mean of %d simulations): " % sims + "  ".join(f"{k} {null[k].mean():.3f}" for k in ["auc", "ap", "hit10", "hit15", "pct"]))
print("\nMEAN OVER SPLITS (sd across splits) | permutation p = P(random >= observed, mean over the same number of splits)")
rows = []
for m in order:
    d = df[df.method == m]; r = {"method": m}
    for k in ["auc", "ap", "hit10", "hit15", "pct"]:
        r[k] = f"{d[k].mean():.3f} ({d[k].std():.3f})"
    r["p(auc)"] = f"{(np.sum(null['auc'] >= d.auc.mean()) + 1) / (sims + 1):.3f}"; r["p(ap)"] = f"{(np.sum(null['ap'] >= d.ap.mean()) + 1) / (sims + 1):.3f}"
    rows.append(r)
print(pd.DataFrame(rows).set_index("method").to_string())
print("\nAUC by split:"); print(df.pivot(index="method", columns="split", values="auc").reindex(order).round(2).to_string())
print("\nCLASSIFICATION VIEW (flag top 10%% = %d of %d), mean over splits" % (round(.1 * n_te), n_te))
print(cdf.groupby("method")[["acc", "prec", "rec", "bal"]].mean().reindex(order).round(3).to_string())
print(f"trivial 'nothing is fraud': accuracy {1-P/n_te:.3f}, recall 0, balanced accuracy 0.500; random flagging recall {null['top10pct_rec'].mean():.3f}")
print("\nPER-FRAUD mean percentile rank among test transactions (0 = top, 0.5 = random), averaged over splits where it was held out:")
for name in ["reservoir_rule", "learned_all"]:
    print(" ", name, {t: round(float(np.mean(v)), 2) for (nm, t), v in sorted(per_fraud.items()) if nm == name})
df.to_csv(f"{od}/fraud_scores_by_split.csv", index=False)
