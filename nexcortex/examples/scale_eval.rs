//! Scalability + sanity run on a synthetic planted-community graph of any size.
//! usage: scale_eval <entities> <avg_degree> [queries=20] [community=20]
//! Reports build time per stage, peak RSS, per-query time, and precision@10 for recovering the rest of the
//! evidence's community (ground truth known), next to the exact random expectation.
use anyhow::{Context, Result};
use nexcortex::*;
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unif(&mut self) -> f64 {
        (self.next() as f64) / ((1u64 << 31) as f64)
    }
}
fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l| l.starts_with("VmHWM")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())).unwrap_or(0.0) / 1024.0
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).context("usage: scale_eval <entities> <avg_degree> [queries] [community]")?.parse()?;
    let deg: f64 = a.get(2).context("avg_degree")?.parse()?;
    let nq: usize = a.get(3).map(|x| x.parse()).transpose()?.unwrap_or(20);
    let cs: usize = a.get(4).map(|x| x.parse()).transpose()?.unwrap_or(20);
    let comm = |i: usize| i / cs;
    let ncomm = (n + cs - 1) / cs;
    let mut rng = Lcg(42);
    // 70% of each node's ties inside its community, 30% across (so the structure is recoverable but noisy)
    let mut edges = std::collections::HashSet::new();
    let target = (n as f64 * deg / 2.0) as usize;
    while edges.len() < target {
        let i = (rng.next() as usize) % n;
        let j = if rng.unif() < 0.7 {
            let lo = comm(i) * cs;
            lo + (rng.next() as usize) % cs.min(n - lo)
        } else {
            (rng.next() as usize) % n
        };
        if i != j {
            edges.insert((i.min(j), i.max(j)));
        }
    }
    let mut b = KbBuilder::new();
    for i in 0..n {
        b = b.entity(&format!("e{i}"), &format!("c{}", comm(i) % 8));
    }
    // a ring through all entities guarantees one connected component
    for i in 0..n {
        edges.insert((i.min((i + 1) % n), i.max((i + 1) % n)));
    }
    let mut ev: Vec<_> = edges.iter().copied().collect();
    ev.sort();
    for (i, j) in &ev {
        b = b.relate(&format!("e{i}"), &format!("e{j}"), "tie", 1.0);
    }
    let t = Instant::now();
    let kb = b.build()?;
    println!("n={n} ties={} (avg degree {:.1}) | kb built {:.1}s | rss {:.0} MB", ev.len(), 2.0 * ev.len() as f64 / n as f64, t.elapsed().as_secs_f64(), peak_rss_mb());
    let t = Instant::now();
    let cfg = PipelineConfig { top_k: 10, rerank_pool: 10, prior_weight: 0.0, ..PipelineConfig::default() };
    let pipe = Pipeline::new(kb, cfg)?;
    println!("pipeline built {:.1}s | peak rss {:.0} MB", t.elapsed().as_secs_f64(), peak_rss_mb());
    let (mut hits, mut tot, mut tq) = (0.0, 0.0, 0.0);
    let mut rand_p = 0.0;
    for q in 0..nq {
        let c = (rng.next() as usize) % ncomm;
        let members: Vec<usize> = (c * cs..((c + 1) * cs).min(n)).collect();
        if members.len() < 8 {
            continue;
        }
        let evid = &members[..3];
        let query = Query {
            question: format!("q{q}"),
            events: evid.iter().map(|i| EventSpec { entity: format!("e{i}"), severity: 0.8, note: String::new() }).collect(),
        };
        let t = Instant::now();
        let an = pipe.analyze(&query)?;
        tq += t.elapsed().as_secs_f64();
        let hidden: std::collections::HashSet<String> = members[3..].iter().map(|i| format!("e{i}")).collect();
        let top: Vec<&str> = an.brief.candidates.iter().map(|c| c.entity.as_str()).collect();
        hits += top.iter().filter(|e| hidden.contains(**e)).count() as f64;
        tot += top.len() as f64;
        rand_p += hidden.len() as f64 / (n - 3) as f64;
    }
    println!("{nq} queries: {:.2}s/query | precision@10 {:.3} (random {:.4}) | peak rss {:.0} MB", tq / nq as f64, hits / tot, rand_p / nq as f64, peak_rss_mb());
    Ok(())
}
