use nexgraph::{props, Database};
use std::time::Instant;
fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let t0 = Instant::now();
    let nodes_txt = std::fs::read_to_string(format!("{dir}/nodes.csv")).unwrap();
    let edges_txt = std::fs::read_to_string(format!("{dir}/edges.csv")).unwrap();
    println!("read files      {:?}", t0.elapsed());
    let t = Instant::now();
    let pn: Vec<i64> = nodes_txt.lines().skip(1).map(|l| l.parse().unwrap()).collect();
    let pe: Vec<(usize, usize)> = edges_txt.lines().skip(1).map(|l| { let (a, b) = l.split_once(',').unwrap(); (a.parse().unwrap(), b.parse().unwrap()) }).collect();
    println!("parse           {:?}", t.elapsed());
    let db = Database::in_memory();
    let t = Instant::now();
    let mut tx = db.begin();
    tx.create_property_index("N", "k");
    let nodes: Vec<u64> = pn.iter().map(|&k| tx.create_node(&["N"], props([("k", k)])).unwrap()).collect();
    println!("build node ops  {:?}", t.elapsed());
    let t = Instant::now();
    for &(a, b) in &pe { tx.create_edge("E", nodes[a], nodes[b], 1.0, Default::default()).unwrap(); }
    println!("build edge ops  {:?}", t.elapsed());
    let t = Instant::now();
    tx.commit().unwrap();
    println!("commit          {:?}", t.elapsed());
}
