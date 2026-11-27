"""Relational baselines (edge table + index, no graph engine): run_sql.py duckdb|sqlite DATA OUT.json
Shortest path is not run: a recursive-CTE BFS is not a like-for-like workload."""
import json, sys, time, sqlite3
eng, d, outp = sys.argv[1:4]
if eng == "duckdb":
    import duckdb; con = duckdb.connect(":memory:"); ver = "duckdb " + duckdb.__version__
    ex = lambda q, p=(): con.execute(q, list(p))
else:
    con = sqlite3.connect(":memory:"); ver = "sqlite " + sqlite3.sqlite_version
    ex = lambda q, p=(): con.execute(q, list(p))
rd = lambda n: [l.strip() for l in open(f"{d}/{n}").read().splitlines() if l.strip()]
t = time.perf_counter()
ex("CREATE TABLE n(k BIGINT PRIMARY KEY)"); ex("CREATE TABLE e(a BIGINT, b BIGINT)")
nodes = [(int(x),) for x in rd("nodes.csv")[1:]]; edges = [tuple(map(int, l.split(","))) for l in rd("edges.csv")[1:]]
if eng == "duckdb":
    ex(f"COPY n FROM '{d}/nodes.csv' (HEADER)"); ex(f"COPY e FROM '{d}/edges.csv' (HEADER)")
else:
    con.executemany("INSERT INTO n VALUES (?)", nodes); con.executemany("INSERT INTO e VALUES (?,?)", edges)
ex("CREATE INDEX ea ON e(a)")
load_ms = (time.perf_counter() - t) * 1e3; print(f"{eng:10} load {load_ms:.0f} ms")
rows = []
def summarize(name, lat, check):
    lat = sorted(lat); med = lat[len(lat)//2]; p95 = lat[int((len(lat)-1)*.95)]
    print(f"{eng:10} {name:8} n={len(lat):5} median={med:12.1f}us p95={p95:12.1f}us")
    rows.append({"workload": name, "n": len(lat), "median_us": med, "p95_us": p95, "total_ms": sum(lat)/1e3, "checksum": check})
def fetch(q, p=()):
    cur = ex(q, p); return cur.fetchall()
def first(q, p=()):
    r = fetch(q, p); return int(r[0][0]) if r else -1
def run(name, f, q, warm, nparams=1):
    ids = [int(x) for x in rd(f + ".txt")]
    for a in ids[:warm]: first(q, (a,) * nparams)
    lat, res = [], []
    for a in ids:
        t = time.perf_counter(); res.append(first(q, (a,) * nparams)); lat.append((time.perf_counter()-t)*1e6)
    summarize(name, lat, str(res))
ph = "?" if eng == "sqlite" else "?"
lat = []
for _ in range(2000):
    t = time.perf_counter(); fetch("SELECT 1"); lat.append((time.perf_counter()-t)*1e6)
summarize("noop", lat, "[]")
run("point", "point", f"SELECT k FROM n WHERE k = {ph}", 30)
run("hop1", "hop1", f"SELECT count(*) FROM e WHERE a = {ph}", 30)
run("hop2", "hop2", f"SELECT count(*) FROM e e1 JOIN e e2 ON e2.a = e1.b WHERE e1.a = {ph}", 30)
reach = (f"SELECT count(*) FROM (SELECT b FROM e WHERE a={ph} UNION "
         f"SELECT e2.b FROM e e1 JOIN e e2 ON e2.a=e1.b WHERE e1.a={ph} UNION "
         f"SELECT e3.b FROM e e1 JOIN e e2 ON e2.a=e1.b JOIN e e3 ON e3.a=e2.b WHERE e1.a={ph})")
run("reach3", "reach3", reach, 10, 3)
def once(name, q, fmt):
    lat, last = [], None
    for _ in range(3):
        t = time.perf_counter(); last = fetch(q); lat.append((time.perf_counter()-t)*1e6)
    summarize(name, lat, fmt(last))
once("scan", "SELECT count(*) FROM e", lambda r: f"[{r[0][0]}]")
once("degree", "SELECT a AS k, count(*) AS d FROM e GROUP BY a ORDER BY d DESC, k LIMIT 5", lambda r: str([[x[0], x[1]] for x in r]))
once("withagg", "SELECT count(*) FROM (SELECT a, count(b) AS d FROM e GROUP BY a HAVING count(b) > 12)", lambda r: f"[{r[0][0]}]")
json.dump({"system": ver + " (relational, edge table)", "load_ms": load_ms, "pagerank_ms": None, "rows": rows}, open(outp, "w"))
