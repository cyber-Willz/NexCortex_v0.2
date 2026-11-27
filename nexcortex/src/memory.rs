//! Stage 3 — **connectome RNN as working memory and reasoning substrate**.
//!
//! The knowledge graph is turned into a `Connectome` (regions = entities, tie strength = tract
//! weight, stronger tie = shorter fibre = shorter conduction delay) and wrapped in the
//! `human_connectome_rnn` network: Dale's-law E/I units per region, conduction delays, leaky
//! tanh dynamics, spectral radius calibrated below 1.
//!
//! Evidence arrives as a time series of graph signals (the diffusion fields of stage 2), one
//! input channel per entity (identity input matrix), pulse by pulse. After the last pulse the
//! network is left to run **silently**: whatever the recurrent wiring still holds then is the
//! working-memory state, and where it sits is the result of evidence propagating along delayed,
//! excitatory/inhibitory paths rather than of a one-shot diffusion.
//!
//! Readout: the **post-evidence window** — the mean excitatory rate over the `window` steps right
//! after the last pulse. (An end-of-run readout was tried first and rejected: after the transient
//! dies out it reflects only the connectome's dominant mode and is the same for any evidence.)
//! Because the memory fades, later evidence is weighted more than earlier evidence; see
//! `Pipeline`'s order-sensitivity check.
//!
//! Honest scope: the network is used as a fixed, calibrated reservoir. Nothing here trains it
//! (the crate's trainable task has 3 inputs / 2 outputs and does not apply to this readout).

use crate::features::EventField;
use anyhow::{anyhow, bail, Result};
use hcrnn::connectome::Connectome;
use hcrnn::network::{Calibration, ConnectomeRnn, RnnConfig};
use hcrnn::task::Trial;

#[derive(Clone, Debug)]
pub struct MemoryConfig {
    /// Steps between the onsets of consecutive evidence pulses.
    pub event_gap: usize,
    /// Duration of each evidence pulse in steps.
    pub pulse: usize,
    /// Silent steps simulated after the last pulse (must be >= `window`).
    pub settle: usize,
    /// Salience is the mean excitatory rate over the `window` steps right after the last pulse.
    pub window: usize,
    /// Scale of the identity input matrix.
    pub input_gain: f64,
    /// Calibrated spectral radius of the initial recurrent matrix (< 1 = fading memory).
    pub gain: f64,
    pub seed: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig { event_gap: 4, pulse: 6, settle: 30, window: 12, input_gain: 2.0, gain: 0.9, seed: 7 }
    }
}

#[derive(Clone, Debug)]
pub struct MemoryTrace {
    pub steps: usize,
    /// Step at which the last evidence pulse ends.
    pub input_end: usize,
    /// Mean excitatory rate per entity over the post-evidence window (the working-memory state).
    pub salience: Vec<f64>,
    /// Mean excitatory rate per entity over the last `window` steps of evidence input.
    pub at_input_end: Vec<f64>,
    /// L2 norm of the excitatory rates per step (length `steps + 1`).
    pub activity: Vec<f64>,
    /// `activity(input_end + window) / activity(input_end)`; ~0 = evidence forgotten, ~1 = fully held.
    pub retention: f64,
    /// Steps after the input ends until activity halves (None if it never does within `settle`).
    pub half_life: Option<usize>,
    pub spectral_radius: f64,
    pub n_units: usize,
    pub n_connections: usize,
    /// Excitatory rates, `(steps + 1) x n_regions` row-major (row 0 = initial state).
    pub e_rates: Vec<f64>,
    pub n_regions: usize,
}

impl MemoryTrace {
    /// Mean excitatory rate per entity over steps `from..to` (exclusive).
    pub fn mean_between(&self, from: usize, to: usize) -> Vec<f64> {
        let n = self.n_regions;
        let mut m = vec![0.0; n];
        for s in from..to {
            for i in 0..n {
                m[i] += self.e_rates[s * n + i];
            }
        }
        m.iter_mut().for_each(|x| *x /= (to - from).max(1) as f64);
        m
    }
}

pub struct WorkingMemory {
    net: ConnectomeRnn,
    cfg: MemoryConfig,
    n: usize,
}

impl WorkingMemory {
    pub fn new(conn: &Connectome, cfg: MemoryConfig) -> Result<WorkingMemory> {
        let n = conn.n();
        // Readout regions are unused by the salience readout but the network requires some:
        // take the three strongest.
        let mut by_strength: Vec<usize> = (0..n).collect();
        by_strength.sort_by(|&a, &b| conn.strength(b).partial_cmp(&conn.strength(a)).unwrap());
        let rc = RnnConfig {
            calibration: Calibration::Power,
            gain: cfg.gain,
            n_in: n,
            n_out: 1,
            input_regions: (0..n).collect(),
            output_regions: by_strength.into_iter().take(3.min(n)).collect(),
            seed: cfg.seed,
            ..RnnConfig::default()
        };
        let mut net = ConnectomeRnn::new(&conn.scaled_to_mean_strength(1.0), rc).map_err(|e| anyhow!(e))?;
        // Identity input matrix: channel c drives the excitatory unit of region c.
        let w_in = net.layout.w_in;
        for k in 0..n * n {
            net.params[w_in + k] = 0.0;
        }
        for i in 0..n {
            net.params[w_in + i * n + i] = cfg.input_gain;
        }
        net.check_constraints().map_err(|e| anyhow!("constraint check failed: {e}"))?;
        Ok(WorkingMemory { net, cfg, n })
    }

    pub fn network(&self) -> &ConnectomeRnn {
        &self.net
    }

    /// Present the evidence fields in order and let the network settle.
    pub fn run(&self, events: &[EventField]) -> Result<MemoryTrace> {
        let c = &self.cfg;
        if events.is_empty() {
            bail!("no evidence to present");
        }
        let n = self.n;
        let input_end = (events.len() - 1) * c.event_gap + c.pulse;
        let t = input_end + c.settle;
        let mut inputs = vec![0.0; t * n];
        for (k, ev) in events.iter().enumerate() {
            if ev.signal.len() != n {
                bail!("evidence field has {} entries, network has {n} regions", ev.signal.len());
            }
            for s in k * c.event_gap..k * c.event_gap + c.pulse {
                for i in 0..n {
                    inputs[s * n + i] += ev.signal[i];
                }
            }
        }
        let trial = Trial {
            t,
            n_in: n,
            n_out: 1,
            inputs,
            targets: vec![0.0; t],
            mask: vec![0.0; t],
            angle: 0.0,
            cue_end: input_end,
            go: t,
        };
        let ro = self.net.forward(&trial);
        let nu = self.net.n_units;
        let e_rates = |s: usize| &ro.r[s * nu..s * nu + n];
        let mean_over = |from: usize, to: usize| -> Vec<f64> {
            let mut m = vec![0.0; n];
            for s in from..to {
                for (i, x) in e_rates(s).iter().enumerate() {
                    m[i] += x;
                }
            }
            m.iter_mut().for_each(|x| *x /= (to - from) as f64);
            m
        };
        if c.window == 0 || c.window > c.settle {
            bail!("memory window ({}) must be between 1 and settle ({})", c.window, c.settle);
        }
        let w = c.window;
        let salience = mean_over(input_end + 1, input_end + 1 + w);
        let at_input_end = mean_over(input_end + 1 - w.min(input_end), input_end + 1);
        let activity: Vec<f64> = (0..=t).map(|s| e_rates(s).iter().map(|x| x * x).sum::<f64>().sqrt()).collect();
        let a_end = activity[input_end].max(1e-12);
        let retention = activity[input_end + w] / a_end;
        let half_life = (input_end..=t).find(|&s| activity[s] <= 0.5 * a_end).map(|s| s - input_end);
        Ok(MemoryTrace {
            steps: t,
            input_end,
            salience,
            at_input_end,
            activity,
            retention,
            half_life,
            spectral_radius: self.net.spectral_radius(),
            n_units: nu,
            n_connections: self.net.edges.len(),
            e_rates: (0..=t).flat_map(|s| e_rates(s).to_vec()).collect(),
            n_regions: n,
        })
    }
}
