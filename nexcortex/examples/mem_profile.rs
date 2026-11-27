//! Stage-by-stage time / RSS / high-water-mark of pipeline construction on a synthetic graph.
//! usage: mem_profile <entities> <avg_degree>
use anyhow::Result;
use nexcortex::*;
use std::time::Instant;
struct Lcg(u64);
impl Lcg { fn next(&mut self) -> u64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.0 >> 33 } fn unif(&mut self) -> f64 { (self.next() as f64) / ((1u64 << 31) as f64) } }
fn kb_of(f: &str) -> f64 { std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l| l.starts_with(f)).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())).unwrap_or(0.0) / 1024.0 }
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a[1].parse()?; let deg: f64 = a[2].parse()?;
    let mut rng = Lcg(42); let cs = 20usize;
    let mut set = std::collections::HashSet::new();
    while set.len() < (n as f64 * deg / 2.0) as usize {
        let i = (rng.next() as usize) % n;
        let j = if rng.unif() < 0.7 { let lo = i / cs * cs; lo + (rng.next() as usize) % cs.min(n - lo) } else { (rng.next() as usize) % n };
        if i != j { set.insert((i.min(j), i.max(j))); }
    }
    for i in 0..n { set.insert((i.min((i + 1) % n), i.max((i + 1) % n))); }
    let mut ev: Vec<_> = set.into_iter().collect(); ev.sort();
    let mut b = KbBuilder::new();
    for i in 0..n { b = b.entity(&format!("e{i}"), "x"); }
    for (i, j) in &ev { b = b.relate(&format!("e{i}"), &format!("e{j}"), "tie", 1.0); }
    let kb = b.build()?;
    println!("n={n} ties={} n^2*8 = {:.0} MB", ev.len(), (n * n * 8) as f64 / 1048576.0);
    let mut last = kb_of("VmRSS");
    let mut stage = |name: &str, t: Instant| { let r = kb_of("VmRSS"); println!("  {name:<34} {:>7.2}s  rss {:>6.0} MB ({:+.0})  hwm {:>6.0} MB", t.elapsed().as_secs_f64(), r, r - last, kb_of("VmHWM")); last = r; };
    stage("knowledge base built", Instant::now());
    let t = Instant::now(); let _c = kb.components()?; stage("kb.components() [adjacency #1]", t);
    let cfg = PipelineConfig { top_k: 10, rerank_pool: 10, prior_weight: 0.0, ..PipelineConfig::default() };
    let t = Instant::now(); let st = features::structure(&kb, &cfg.features)?; stage("features::structure (spectral)", t);
    let systems: Vec<String> = st.cluster.iter().map(|c| format!("C{c}")).collect();
    let pos: Vec<[f64; 3]> = st.embedding.iter().map(|p| [p[0] * 100.0, p[1] * 100.0, p[2] * 100.0]).collect();
    let t = Instant::now(); let conn = kb.to_connectome(&pos, &systems)?; stage("kb.to_connectome [adjacency #2, len]", t);
    let t = Instant::now(); let wm = memory::WorkingMemory::new(&conn, cfg.memory.clone())?; stage("WorkingMemory::new (reservoir)", t);
    let t = Instant::now(); let adj = kb.adjacency()?; stage("kb.adjacency() [Pipeline keeps it]", t);
    let _ = (&wm, &adj, &conn);
    Ok(())
}
