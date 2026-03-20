use serde::{Deserialize, Serialize};

const MIN_SAMPLES: u64 = 128;
const EPSILON: f64 = 1e-8;
const DRIFT_ALPHA: f64 = 0.05;
const STABILITY_BETA: f64 = 0.999;
const SUSPICION_DECAY: f64 = 0.98;

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Clean,
    Suspect { deviation: f64, reason: &'static str },
    Poison { severity: f64, reason: &'static str },
}

impl Verdict {
    #[inline(always)]
    pub const fn is_fatal(&self) -> bool {
        matches!(self, Verdict::Poison { .. })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerStats {
    mean: f64,
    m2: f64,
    count: u64,
    ewma: f64,
    stable_std: f64,
    suspicion: f64,
}

#[derive(Debug, Default)]
pub struct AnomalyDetector {
    layers: Vec<LayerStats>,
    base_threshold: f64,
}

impl AnomalyDetector {
    pub fn new(base_threshold: f64) -> Self {
        Self {
            layers: Vec::with_capacity(1024),
            base_threshold,
        }
    }

    #[inline(never)]
    pub fn validate(&mut self, layer_id: u32, gradients: &[f32]) -> Verdict {
        if gradients.is_empty() {
            return Verdict::Clean;
        }

        let id = layer_id as usize;
        if id >= self.layers.len() {
            self.layers.resize(id + 1, LayerStats::default());
        }
        let stats = &mut self.layers[id];

        let sum: f32 = gradients.iter().map(|&v| v.abs()).sum();
        let val = (sum / gradients.len() as f32) as f64;

        stats.count += 1;
        let count_f = stats.count as f64;

        let delta = val - stats.mean;
        stats.mean += delta / count_f;
        stats.m2 = delta.mul_add(val - stats.mean, stats.m2);

        if stats.count == 1 {
            stats.ewma = val;
            stats.stable_std = EPSILON;
        } else {
            stats.ewma = DRIFT_ALPHA.mul_add(val, (1.0 - DRIFT_ALPHA) * stats.ewma);
        }

        if stats.count < MIN_SAMPLES {
            stats.suspicion *= SUSPICION_DECAY;
            return Verdict::Clean;
        }

        let stddev = (stats.m2 / (count_f - 1.0)).sqrt();

        stats.stable_std = STABILITY_BETA
            .mul_add(stats.stable_std, (1.0 - STABILITY_BETA) * stddev)
            .max(stddev);

        let effective_std = val.max(stats.stable_std).max(EPSILON);
        let z = ((val - stats.mean) / effective_std).abs();

        if z > self.base_threshold {
            let severity = z - self.base_threshold;
            stats.suspicion = SUSPICION_DECAY.mul_add(stats.suspicion, severity);

            return if severity > 2.0 || stats.suspicion > 3.0 {
                Verdict::Poison { severity: z, reason: "Gradient Spike" }
            } else {
                Verdict::Suspect { deviation: z, reason: "Spike Instability" }
            };
        }

        let drift = (stats.ewma - stats.mean).abs();
        let drift_threshold = effective_std * 2.0;

        if drift > drift_threshold {
            let severity = drift / drift_threshold;
            stats.suspicion = SUSPICION_DECAY.mul_add(stats.suspicion, severity);

            return if severity > 1.5 || stats.suspicion > 3.0 {
                Verdict::Poison { severity, reason: "Sustained Drift" }
            } else {
                Verdict::Suspect { deviation: severity, reason: "Drift Detected" }
            };
        }

        stats.suspicion *= SUSPICION_DECAY;
        Verdict::Clean
    }
}