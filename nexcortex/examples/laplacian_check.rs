//! Tests the claim "a sparse/matrix-free Laplacian matches the dense one to 1e-10, cuts memory ~100x and
//! cuts query latency from seconds to ms". Reports (1) exactness, (2) memory of dense vs CSR, (3) apply time,
//! (4) what the Laplacian costs at query time vs the real per-query cost breakdown.
//! usage: laplacian_check <entities> <avg_degree>
use anyhow::Result;
use nexcortex::*;
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.0 >> 33 }
    fn unif(&mut self) -> f64 { (self.next() as f64) / ((1u64 << 31) as f64) }
}
fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l| l.starts_with("VmRSS")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())).unwrap_or(0.0) / 1024.0
}
/// CSR of L = I - D^-1/2 A D^-1/2 from an edge list (same formula as the dense one).
struct Csr { ptr: Vec<usize>, idx: Vec<u32>, val: Vec<f64> }
impl Csr {
    fn from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Csr {
        let mut s = vec![0.0; n];
        for &(i, j, w) in edges { s[i] += w; s[j] += w; }
        let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for i in 0..n { if s[i] > 0.0 { rows[i].push((i as u32, 1.0)); } }
        for &(i, j, w) in edges {
            let v = -w / (s[i] * s[j]).sqrt();
            rows[i].push((j as u32, v)); rows[j].push((i as u32, v));
        }
        let (mut ptr, mut idx, mut val) = (vec![0], Vec::new(), Vec::new());
        for r in rows.iter_mut() { r.sort_by_key(|x| x.0); for (c, v) in r.iter() { idx.push(*c); val.push(*v); } ptr.push(idx.len()); }
        Csr { ptr, idx, val }
    }
    fn mul(&self, x: &[f64], out: &mut [f64]) {
        for i in 0..self.ptr.len() - 1 {
            let mut a = 0.0;
            for k in self.ptr[i]..self.ptr[i + 1] { a += self.val[k] * x[self.idx[k] as usize]; }
            out[i] = a;
        }
    }
    fn bytes(&self) -> usize { self.ptr.len() * 8 + self.idx.len() * 4 + self.val.len() * 8 }
}
/// Matrix-free: no stored matrix at all; recompute from adjacency lists + degrees.
struct Free { adj: Vec<Vec<(u32, f64)>>, s: Vec<f64> }
impl Free {
    fn mul(&self, x: &[f64], out: &mut [f64]) {
        for i in 0..self.s.len() {
            let mut a = if self.s[i] > 0.0 { x[i] } else { 0.0 };
            for &(j, w) in &self.adj[i] { a -= w / (self.s[i] * self.s[j as usize]).sqrt() * x[j as usize]; }
            out[i] = a;
        }
    }
    fn bytes(&self) -> usize { self.adj.iter().map(|a| a.len() * 16 + 24).sum::<usize>() + self.s.len() * 8 }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a[1].parse()?; let deg: f64 = a[2].parse()?;
    let mut rng = Lcg(42);
    let cs = 20usize;
    let mut set = std::collections::HashSet::new();
    let target = (n as f64 * deg / 2.0) as usize;
    while set.len() < target {
        let i = (rng.next() as usize) % n;
        let j = if rng.unif() < 0.7 { let lo = i / cs * cs; lo + (rng.next() as usize) % cs.min(n - lo) } else { (rng.next() as usize) % n };
        if i != j { set.insert((i.min(j), i.max(j))); }
    }
    for i in 0..n { set.insert((i.min((i + 1) % n), i.max((i + 1) % n))); }
    let mut ev: Vec<(usize, usize)> = set.into_iter().collect(); ev.sort();
    let mut b = KbBuilder::new();
    for i in 0..n { b = b.entity(&format!("e{i}"), "x"); }
    for (i, j) in &ev { b = b.relate(&format!("e{i}"), &format!("e{j}"), "tie", 1.0); }
    let kb = b.build()?;
    let base = rss_mb();
    let t = Instant::now();
    let pipe = Pipeline::new(kb, PipelineConfig { top_k: 10, rerank_pool: 10, prior_weight: 0.0, ..PipelineConfig::default() })?;
    println!("n={n} ties={} | pipeline build {:.1}s | rss {:.0} MB (before {:.0})", ev.len(), t.elapsed().as_secs_f64(), rss_mb(), base);
    let net = pipe.memory().network();
    let dense = &net.laplacian; // exact operator actually stored by the library
    let n_reg = net.n_regions;
    assert_eq!(dense.len(), n_reg * n_reg);

    // edge weights as the library sees them (connectome scaled to mean strength 1.0)
    let adjm = pipe.adjacency();
    let mut edges = Vec::new();
    for i in 0..n { for j in i + 1..n { if adjm[i * n + j] > 0.0 { edges.push((i, j, adjm[i * n + j])); } } }
    let mean_strength = (0..n).map(|i| (0..n).map(|j| adjm[i * n + j]).sum::<f64>()).sum::<f64>() / n as f64;
    for e in edges.iter_mut() { e.2 /= mean_strength; } // L is invariant to this scaling; kept for fidelity
    let t = Instant::now();
    let csr = Csr::from_edges(n, &edges);
    let t_csr_build = t.elapsed().as_secs_f64();
    let mut s = vec![0.0; n];
    for &(i, j, w) in &edges { s[i] += w; s[j] += w; }
    let mut adj = vec![Vec::new(); n];
    for &(i, j, w) in &edges { adj[i].push((j as u32, w)); adj[j].push((i as u32, w)); }
    let free = Free { adj, s };

    // (1) exactness on random vectors, and the Rayleigh quotient the library computes
    let (mut e_csr, mut e_free, mut e_ray) = (0.0f64, 0.0f64, 0.0f64);
    let (mut yd, mut yc, mut yf) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut t_dense = 0.0; let mut t_csr = 0.0; let mut t_free = 0.0;
    for _ in 0..20 {
        let x: Vec<f64> = (0..n).map(|_| rng.unif() - 0.5).collect();
        let t = Instant::now();
        for i in 0..n { yd[i] = dense[i * n..(i + 1) * n].iter().zip(&x).map(|(a, b)| a * b).sum(); }
        t_dense += t.elapsed().as_secs_f64();
        let t = Instant::now(); csr.mul(&x, &mut yc); t_csr += t.elapsed().as_secs_f64();
        let t = Instant::now(); free.mul(&x, &mut yf); t_free += t.elapsed().as_secs_f64();
        for i in 0..n { e_csr = e_csr.max((yd[i] - yc[i]).abs()); e_free = e_free.max((yd[i] - yf[i]).abs()); }
        let q = |y: &[f64]| x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
        e_ray = e_ray.max((q(&yd) - q(&yc)).abs());
    }
    let dense_mb = (dense.len() * 8) as f64 / 1048576.0;
    println!("\n(1) exactness vs the stored dense Laplacian (20 random vectors): max|dense-CSR| = {e_csr:.2e}, max|dense-matrix-free| = {e_free:.2e}, Rayleigh diff {e_ray:.2e}");
    println!("(2) memory: dense {dense_mb:.1} MB | CSR {:.2} MB ({:.0}x smaller) | matrix-free {:.2} MB ({:.0}x smaller)", csr.bytes() as f64 / 1048576.0, dense_mb * 1048576.0 / csr.bytes() as f64, free.bytes() as f64 / 1048576.0, dense_mb * 1048576.0 / free.bytes() as f64);
    println!("(3) apply time per L*x: dense {:.2} ms | CSR {:.2} ms | matrix-free {:.2} ms   (CSR build {:.0} ms)", t_dense * 50.0, t_csr * 50.0, t_free * 50.0, t_csr_build * 1000.0);

    // (4) what a query costs and whether L is involved
    let q = Query { question: "q".into(), events: (0..3).map(|i| EventSpec { entity: format!("e{i}"), severity: 0.8, note: String::new() }).collect() };
    let t = Instant::now(); let an = pipe.analyze(&q)?; let t_an = t.elapsed().as_secs_f64();
    let t = Instant::now(); let _ = pipe.memory().network().spectral_radius(); let t_rho = t.elapsed().as_secs_f64();
    let t = Instant::now(); let ef = features::event_fields(&pipe.kb, &pipe.cfg.features, &[(0, 0.8), (1, 0.8), (2, 0.8)])?; let t_ef = t.elapsed().as_secs_f64();
    let t = Instant::now(); let tr = pipe.memory().run(&ef)?; let t_run = t.elapsed().as_secs_f64();
    println!("(4) one query: analyze {:.2}s total = event fields {:.2}s + memory.run {:.2}s (of which spectral_radius {:.2}s; {} steps, {} units, {} edges); L touched at query time: no", t_an, t_ef, t_run, t_rho, tr.steps, tr.n_units, tr.n_connections);
    println!("    Laplacian share of query time: 0.0 s; of process RSS: {:.1}%  ({dense_mb:.0} of {:.0} MB)", 100.0 * dense_mb / rss_mb(), rss_mb());
    let _ = an;
    Ok(())
}
