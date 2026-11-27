//! Out-of-core microbenchmark: what does memory-mapping buy for the structures this system actually holds?
//! Matrix-vector products (the per-step reservoir workload) on
//!   dense  U x U f64 matrix, mmap'd file        (what the dense layout would need out-of-core)
//!   CSR    U x U with `nnz` entries, in RAM     (sparse in memory)
//!   CSR    mmap'd file, warm and COLD (page cache dropped with posix_fadvise(DONTNEED))
//! usage: ooc_bench <units> <nnz> <steps> [skip_dense=0]
use anyhow::Result;
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

struct Lcg(u64);
impl Lcg { fn next(&mut self) -> u64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.0 >> 33 } }
fn rss() -> f64 { std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l| l.starts_with("VmRSS")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())).unwrap_or(0.0) / 1024.0 }
fn drop_cache(f: &File) { unsafe { libc::posix_fadvise(std::os::unix::io::AsRawFd::as_raw_fd(f), 0, 0, libc::POSIX_FADV_DONTNEED); } }

fn csr_mul(ptr: &[u64], idx: &[u32], val: &[f64], x: &[f64], y: &mut [f64]) {
    for i in 0..ptr.len() - 1 {
        let mut a = 0.0;
        for k in ptr[i] as usize..ptr[i + 1] as usize { a += val[k] * x[idx[k] as usize]; }
        y[i] = a;
    }
}
fn cast<T>(b: &[u8]) -> &[T] { unsafe { std::slice::from_raw_parts(b.as_ptr() as *const T, b.len() / std::mem::size_of::<T>()) } }

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let u: usize = a[1].parse()?; let nnz: usize = a[2].parse()?; let steps: usize = a[3].parse()?;
    let skip_dense = a.get(4).map_or(false, |x| x == "1");
    let dir = std::env::temp_dir().join("ooc_bench"); std::fs::create_dir_all(&dir)?;
    let mut rng = Lcg(7);
    // ---- CSR file: [ptr: u64 x (u+1)][idx: u32 x nnz][val: f64 x nnz]
    let per_row = nnz / u;
    let mut ptr = vec![0u64; u + 1];
    let mut idx = Vec::with_capacity(nnz); let mut val = Vec::with_capacity(nnz);
    for i in 0..u { for _ in 0..per_row { idx.push((rng.next() as usize % u) as u32); val.push((rng.next() % 1000) as f64 / 1e5); } ptr[i + 1] = idx.len() as u64; }
    let real_nnz = idx.len();
    let csr_path = dir.join("csr.bin");
    { let mut w = BufWriter::new(File::create(&csr_path)?);
      w.write_all(bytes(&ptr))?; w.write_all(bytes(&idx))?; w.write_all(bytes(&val))?; w.flush()?; }
    let x: Vec<f64> = (0..u).map(|i| ((i % 17) as f64) / 17.0).collect();
    let mut y = vec![0.0; u];
    let csr_mb = (ptr.len() * 8 + real_nnz * 12) as f64 / 1048576.0;
    println!("units {u}, nnz {real_nnz} (density {:.4}%), CSR file {csr_mb:.1} MB, dense would be {:.0} MB", 100.0 * real_nnz as f64 / (u as f64 * u as f64), (u * u * 8) as f64 / 1048576.0);

    // reference result (RAM CSR)
    let t = Instant::now(); for _ in 0..steps { csr_mul(&ptr, &idx, &val, &x, &mut y); } let t_ram = t.elapsed().as_secs_f64() / steps as f64;
    let y_ref = y.clone();
    println!("CSR in RAM          : {:8.3} ms/step", t_ram * 1e3);

    // mmap CSR
    let f = File::open(&csr_path)?; let m: Mmap = unsafe { MmapOptions::new().map(&f)? };
    let (pb, rest) = m.split_at((u + 1) * 8); let (ib, vb) = rest.split_at(real_nnz * 4);
    let (mp, mi, mv) = (cast::<u64>(pb), cast::<u32>(ib), cast::<f64>(vb));
    drop_cache(&f); unsafe { libc::madvise(m.as_ptr() as *mut _, m.len(), libc::MADV_DONTNEED); }
    let t = Instant::now(); csr_mul(mp, mi, mv, &x, &mut y); let cold = t.elapsed().as_secs_f64();
    let t = Instant::now(); for _ in 0..steps { csr_mul(mp, mi, mv, &x, &mut y); } let warm = t.elapsed().as_secs_f64() / steps as f64;
    let diff = y.iter().zip(&y_ref).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
    println!("CSR mmap, cold 1st  : {:8.3} ms (first step after dropping the page cache)\nCSR mmap, warm      : {:8.3} ms/step   max|diff vs RAM| {:.1e}   rss {:.0} MB", cold * 1e3, warm * 1e3, diff, rss());

    if skip_dense { return Ok(()); }
    // ---- dense file U x U f64
    let dpath = dir.join("dense.bin");
    { let mut w = BufWriter::with_capacity(1 << 22, File::create(&dpath)?); let mut row = vec![0f64; u];
      for _ in 0..u { for v in row.iter_mut() { *v = (rng.next() % 1000) as f64 / 1e5; } w.write_all(bytes(&row))?; } w.flush()?; }
    let f = File::open(&dpath)?; let dm: Mmap = unsafe { MmapOptions::new().map(&f)? }; let d = cast::<f64>(&dm);
    let dense_mul = |x: &[f64], y: &mut [f64]| { for i in 0..u { y[i] = d[i * u..(i + 1) * u].iter().zip(x).map(|(a, b)| a * b).sum(); } };
    drop_cache(&f); unsafe { libc::madvise(dm.as_ptr() as *mut _, dm.len(), libc::MADV_DONTNEED); }
    let t = Instant::now(); dense_mul(&x, &mut y); let dcold = t.elapsed().as_secs_f64();
    let t = Instant::now(); for _ in 0..steps.min(3) { dense_mul(&x, &mut y); } let dwarm = t.elapsed().as_secs_f64() / steps.min(3) as f64;
    println!("dense mmap, cold 1st: {:8.1} ms (reads {:.0} MB)\ndense mmap, warm    : {:8.1} ms/step   rss {:.0} MB", dcold * 1e3, (u * u * 8) as f64 / 1048576.0, dwarm * 1e3, rss());
    Ok(())
}
fn bytes<T>(v: &[T]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<T>()) } }
