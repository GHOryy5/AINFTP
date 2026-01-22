use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use serde::{Serialize, Deserialize};

const MIN_SAMPLES: u64 = 128;
const EPSILON: f64 = 1e-8;
const DRIFT_ALPHA: f64 = 0.05;     // Fast reaction
const STABILITY_BETA: f64 = 0.999; // Slow memory
const SUSPICION_DECAY: f64 = 0.98; // Temporal memory

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<fnv::FnvHasher>>;

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Clean,
    Suspect { deviation: f64, reason: &'static str },
    Poison  { severity: f64, reason: &'static str },
}

impl Verdict {
    #[inline]
    pub fn is_fatal(&self) -> bool {
        matches!(self, Verdict::Poison { .. })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStats {

    mean: f64,
    m2: f64,
    count: u64,

    ewma: f64,

    stable_std: f64,

    suspicion_ewma: f64,
}

impl LayerStats {
    fn new() -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
            ewma: 0.0,
            stable_std: 0.0,
            suspicion_ewma: 0.0,
        }
    }

    fn update(&mut self, val: f64) -> GradientSnapshot {
        self.count += 1;

        let delta = val - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = val - self.mean;
        self.m2 += delta * delta2;

        self.ewma = if self.count == 1 {
            val
        } else {
            DRIFT_ALPHA * val + (1.0 - DRIFT_ALPHA) * self.ewma
        };

        let variance = if self.count > 1 {
            self.m2 / (self.count - 1) as f64
        } else {
            0.0
        };

        let stddev = variance.sqrt();

        // Stability floor
        if self.count > MIN_SAMPLES {
            self.stable_std =
                STABILITY_BETA * self.stable_std +
                (1.0 - STABILITY_BETA) * stddev;
        } else {
            self.stable_std = stddev;
        }

        GradientSnapshot {
            mean: self.mean,
            stddev,
            drift: (self.ewma - self.mean).abs(),
        }
    }

    #[inline]
    fn effective_std(&self, current: f64) -> f64 {
        current.max(self.stable_std).max(EPSILON)
    }
}

#[derive(Debug)]
struct GradientSnapshot {
    mean: f64,
    stddev: f64,
    drift: f64,
}

#[derive(Debug)]
pub struct AnomalyDetector {
    layers: FastMap<u32, LayerStats>,
    base_threshold: f64,
}

impl AnomalyDetector {
    pub fn new(base_threshold: f64) -> Self {
        Self {
            layers: FastMap::with_capacity_and_hasher(1024, Default::default()),
            base_threshold,
        }
    }

    #[inline(never)]
    pub fn validate(&mut self, layer_id: u32, gradients: &[f32]) -> Verdict {
        if gradients.is_empty() {
            return Verdict::Clean;
        }

        let mut sum_mag = 0.0;
        let chunks = gradients.chunks_exact(8);
        let remainder = chunks.remainder();

        for c in chunks {
            sum_mag += (c[0].abs() + c[1].abs() + c[2].abs() + c[3].abs() +
                        c[4].abs() + c[5].abs() + c[6].abs() + c[7].abs()) as f64;
        }
        for v in remainder {
            sum_mag += v.abs() as f64;
        }

        let avg_mag = sum_mag / gradients.len() as f64;

        let stats = self.layers
            .entry(layer_id)
            .or_insert_with(LayerStats::new);

        let snap = stats.update(avg_mag);

        if stats.count < MIN_SAMPLES || snap.stddev < EPSILON {
            stats.suspicion_ewma *= SUSPICION_DECAY;
            return Verdict::Clean;
        }

        let effective_std = stats.effective_std(snap.stddev);
        let z = (avg_mag - snap.mean) / effective_std;

        if z.abs() > self.base_threshold {
            let severity = z.abs() - self.base_threshold;
            stats.suspicion_ewma = stats.suspicion_ewma * SUSPICION_DECAY + severity;

            return if severity > 2.0 || stats.suspicion_ewma > 3.0 {
                Verdict::Poison { severity: z.abs(), reason: "Gradient Spike" }
            } else {
                Verdict::Suspect { deviation: z.abs(), reason: "Spike Instability" }
            };
        }

        let drift_threshold = effective_std * 2.0;
        if snap.drift > drift_threshold {
            let severity = snap.drift / drift_threshold;
            stats.suspicion_ewma = stats.suspicion_ewma * SUSPICION_DECAY + severity;

            return if severity > 1.5 || stats.suspicion_ewma > 3.0 {
                Verdict::Poison { severity, reason: "Sustained Drift" }
            } else {
                Verdict::Suspect { deviation: severity, reason: "Drift Detected" }
            };
        }

        stats.suspicion_ewma *= SUSPICION_DECAY;
        Verdict::Clean
    }
}