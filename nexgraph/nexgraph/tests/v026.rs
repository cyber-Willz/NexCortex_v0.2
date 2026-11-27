//! v0.26: the compact-metadata fast paths in `algo.shortestPath` and variable-length expansion must give exactly the
//! results of the general (body-reading) semantics, on a graph mixing binary edges, directed hyperedges, undirected
//! hyperedges, self-loops, parallel edges, several labels and deleted edges.
use nexgraph::{props, Database, Value};
use std::collections::{HashSet, VecDeque};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[derive(Clone)]
struct E {
    label: &'static str,
    tail: Vec<usize>,
    head: Vec<usize>, // empty = undirected hyperedge over `tail`
    alive: bool,
    id: u64,
}

fn build(n: usize, seed: u64, with_undirected: bool) -> (Database, Vec<E>) {
    let db = Database::in_memory();
    let mut r = Rng(seed);
    let mut tx = db.begin();
    let nodes: Vec<u64> = (0..n).map(|i| tx.create_node(&["N"], props([("k", i as i64)])).unwrap()).collect();
    let mut es: Vec<E> = Vec::new();
    for _ in 0..170 {
        let (a, b) = (r.below(n), r.below(n)); // self-loops and parallel edges happen
        let label = if r.below(4) == 0 { "F" } else { "E" };
        let id = tx.create_edge(label, nodes[a], nodes[b], 1.0, Default::default()).unwrap();
        es.push(E { label, tail: vec![a], head: vec![b], alive: true, id });
    }
    for _ in 0..12 {
        let tail: Vec<usize> = (0..1 + r.below(3)).map(|_| r.below(n)).collect();
        let head: Vec<usize> = (0..1 + r.below(3)).map(|_| r.below(n)).collect();
        let tl: Vec<u64> = tail.iter().map(|&i| nodes[i]).collect();
        let hd: Vec<u64> = head.iter().map(|&i| nodes[i]).collect();
        let id = tx.create_hyperedge("E", &tl, &hd, 1.0, Default::default()).unwrap();
        es.push(E { label: "E", tail, head, alive: true, id });
    }
    if with_undirected {
        for _ in 0..6 {
            let mut m: Vec<usize> = (0..2 + r.below(3)).map(|_| r.below(n)).collect();
            m.sort_unstable();
            m.dedup();
            if m.len() < 2 {
                continue;
            }
            let tl: Vec<u64> = m.iter().map(|&i| nodes[i]).collect();
            let id = tx.create_hyperedge("U", &tl, &[], 1.0, Default::default()).unwrap();
            es.push(E { label: "U", tail: m, head: vec![], alive: true, id });
        }
    }
    tx.commit().unwrap();
    // delete a slice of edges in a later transaction (dead ids stay in the adjacency lists)
    let mut tx = db.begin();
    for e in es.iter_mut() {
        if r.below(9) == 0 {
            tx.delete(e.id);
            e.alive = false;
        }
    }
    tx.commit().unwrap();
    (db, es)
}

fn ref_path(es: &[E], n: usize, a: usize, b: usize) -> Option<usize> {
    let mut dist = vec![usize::MAX; n];
    dist[a] = 0;
    let mut q = VecDeque::from([a]);
    while let Some(u) = q.pop_front() {
        for e in es.iter().filter(|e| e.alive) {
            let nxt: Vec<usize> = if e.head.is_empty() {
                if e.tail.contains(&u) { e.tail.iter().copied().filter(|&x| x != u).collect() } else { vec![] }
            } else if e.tail.contains(&u) {
                e.head.clone()
            } else {
                vec![]
            };
            for v in nxt {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    q.push_back(v);
                }
            }
        }
    }
    (dist[b] != usize::MAX).then_some(dist[b])
}

fn ref_reach(es: &[E], a: usize, label: &str, lo: usize, hi: usize) -> usize {
    fn go(es: &[E], cur: usize, depth: usize, lo: usize, hi: usize, label: &str, used: &mut Vec<usize>, out: &mut HashSet<usize>) {
        if depth >= lo {
            out.insert(cur);
        }
        if depth == hi {
            return;
        }
        for (i, e) in es.iter().enumerate() {
            if !e.alive || e.label != label || e.head.is_empty() || used.contains(&i) || !e.tail.contains(&cur) {
                continue;
            }
            let mut hs = e.head.clone();
            hs.sort_unstable();
            hs.dedup();
            for h in hs {
                used.push(i);
                go(es, h, depth + 1, lo, hi, label, used, out);
                used.pop();
            }
        }
    }
    let mut out = HashSet::new();
    go(es, a, 0, lo, hi, label, &mut Vec::new(), &mut out);
    out.len()
}

fn check(with_undirected: bool, seed: u64) {
    let n = 60;
    let (db, es) = build(n, seed, with_undirected);
    let mut r = Rng(seed ^ 0x9e3779b97f4a7c15);
    for _ in 0..250 {
        let (a, b) = (r.below(n), r.below(n));
        let q = db.query(&format!("MATCH (a:N {{k:{a}}}), (b:N {{k:{b}}}) CALL algo.shortestPath(a, b) YIELD length RETURN length")).unwrap();
        let got = q.rows.first().map(|x| match &x[0] {
            Value::Int(i) => *i as usize,
            o => panic!("{o:?}"),
        });
        assert_eq!(got, ref_path(&es, n, a, b), "shortest path {a}->{b} (undirected={with_undirected}, seed={seed})");
    }
    for a in 0..n {
        for (label, lo, hi) in [("E", 1, 3), ("E", 2, 3), ("F", 1, 2), ("E", 0, 2)] {
            let q = db.query(&format!("MATCH (a:N {{k:{a}}})-[:{label}*{lo}..{hi}]->(b) RETURN count(DISTINCT b)")).unwrap();
            let got = match &q.rows[0][0] {
                Value::Int(i) => *i as usize,
                o => panic!("{o:?}"),
            };
            assert_eq!(got, ref_reach(&es, a, label, lo, hi), "reach {a} -[:{label}*{lo}..{hi}]-> (seed={seed})");
        }
    }
}

#[test]
fn fast_paths_match_reference_on_directed_graphs() {
    for seed in [1, 2, 3] {
        check(false, seed);
    }
}

#[test]
fn fast_paths_match_reference_with_undirected_hyperedges() {
    for seed in [4, 5, 6] {
        check(true, seed);
    }
}
