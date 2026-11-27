"""FalkorDB (embedded via falkordblite) driver, same workloads/checksums as run_kuzu.py: run_falkor.py DATA OUT.json"""
import json, sys, time, tempfile, os
from redislite import FalkorDB
d, outp = sys.argv[1:3]
tmp = tempfile.mkdtemp(); db = FalkorDB(os.path.join(tmp, "f.db")); g = db.select_graph("g")
rd = lambda n: [l.strip() for l in open(f"{d}/{n}").read().splitlines() if l.strip()]
t = time.perf_counter()
g.query("CREATE INDEX FOR (n:N) ON (n.k)")
nodes = [int(x) for x in rd("nodes.csv")[1:]]
for i in range(0, len(nodes), 10000): g.query("UNWIND $ks AS k CREATE (:N {k:k})", {"ks": nodes[i:i+10000]})
edges = [list(map(int, l.split(","))) for l in rd("edges.csv")[1:]]
for i in range(0, len(edges), 5000): g.query("UNWIND $es AS e MATCH (a:N {k:e[0]}), (b:N {k:e[1]}) CREATE (a)-[:E]->(b)", {"es": edges[i:i+5000]})
load_ms = (time.perf_counter() - t) * 1e3; print(f"falkordb   load {load_ms:.0f} ms")
rows = []
def summarize(name, lat, check):
    lat = sorted(lat); med = lat[len(lat)//2]; p95 = lat[int((len(lat)-1)*.95)]
    print(f"falkordb   {name:8} n={len(lat):5} median={med:12.1f}us p95={p95:12.1f}us")
    rows.append({"workload": name, "n": len(lat), "median_us": med, "p95_us": p95, "total_ms": sum(lat)/1e3, "checksum": check})
def first(q, p=None):
    r = g.query(q, p).result_set
    return int(r[0][0]) if r else -1
def run(name, f, q, warm):
    ids = [int(x) for x in rd(f + ".txt")]
    for a in ids[:warm]: first(q, {"k": a})
    lat, res = [], []
    for a in ids:
        t = time.perf_counter(); res.append(first(q, {"k": a})); lat.append((time.perf_counter()-t)*1e6)
    summarize(name, lat, str(res))
lat = []
for _ in range(2000):
    t = time.perf_counter(); g.query("RETURN 1"); lat.append((time.perf_counter()-t)*1e6)
summarize("noop", lat, "[]")
run("point", "point", "MATCH (a:N {k: $k}) RETURN a.k", 30)
run("hop1", "hop1", "MATCH (a:N {k: $k})-[:E]->(b) RETURN count(b)", 30)
run("hop2", "hop2", "MATCH (a:N {k: $k})-[:E]->(b)-[:E]->(c) RETURN count(*)", 30)
run("reach3", "reach3", "MATCH (a:N {k: $k})-[:E*1..3]->(b) RETURN count(DISTINCT b)", 10)
def once(name, q, fmt):
    lat, last = [], None
    for _ in range(3):
        t = time.perf_counter(); last = g.query(q).result_set; lat.append((time.perf_counter()-t)*1e6)
    summarize(name, lat, fmt(last))
once("scan", "MATCH (a:N)-[:E]->(b) RETURN count(*)", lambda r: f"[{r[0][0]}]")
once("degree", "MATCH (a:N)-[:E]->(b) RETURN a.k AS k, count(*) AS d ORDER BY d DESC, k LIMIT 5", lambda r: str([[x[0], x[1]] for x in r]))
once("withagg", "MATCH (a:N)-[:E]->(b) WITH a, count(b) AS d WHERE d > 12 RETURN count(*)", lambda r: f"[{r[0][0]}]")
pairs = [tuple(map(int, l.split())) for l in rd("pairs.txt")]; lat, res = [], []
sp = "MATCH (a:N {k:$a}), (b:N {k:$b}) RETURN length(shortestPath((a)-[:E*..12]->(b)))"
for i, (a, b) in enumerate(pairs):
    t = time.perf_counter(); v = first(sp, {"a": a, "b": b})
    if i >= 5: lat.append((time.perf_counter()-t)*1e6); res.append(v)
summarize("path", lat, str(res))
json.dump({"system": "falkordb (falkordblite 0.12.2)", "load_ms": load_ms, "pagerank_ms": None, "rows": rows}, open(outp, "w"))
db.close() if hasattr(db, "close") else None
