use nexgraph::{props, Database};
use std::time::Instant;
fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let db = Database::in_memory();
    let mut tx = db.begin();
    tx.create_property_index("N", "k");
    let nodes: Vec<u64> = std::fs::read_to_string(format!("{dir}/nodes.csv")).unwrap().lines().skip(1).map(|l| tx.create_node(&["N"], props([("k", l.parse::<i64>().unwrap())])).unwrap()).collect();
    for l in std::fs::read_to_string(format!("{dir}/edges.csv")).unwrap().lines().skip(1) {
        let (a, b) = l.split_once(',').unwrap();
        tx.create_edge("E", nodes[a.parse::<usize>().unwrap()], nodes[b.parse::<usize>().unwrap()], 1.0, Default::default()).unwrap();
    }
    tx.commit().unwrap();
    for (n, q) in [("scan", "MATCH (a:N)-[:E]->(b) RETURN count(*)"), ("degree", "MATCH (a:N)-[:E]->(b) RETURN a.k AS k, count(*) AS d ORDER BY d DESC, k LIMIT 5"), ("withagg", "MATCH (a:N)-[:E]->(b) WITH a, count(b) AS d WHERE d > 12 RETURN count(*)")] {
        let mut v = vec![];
        for _ in 0..8 { let t = Instant::now(); db.query(q).unwrap(); v.push(t.elapsed().as_secs_f64() * 1e3); }
        println!("{n:8} {}", v.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(" "));
    }
}
