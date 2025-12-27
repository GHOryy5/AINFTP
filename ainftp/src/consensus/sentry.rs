//! Anomaly Sentry
//! 
//! Uses statistical analysis to detect "Poison Gradients".
//! 
//! The Problem: If a node in the cluster is malicious or buggy, it might send 
//! gradients that are 10x larger than normal (e.g., weight = 1,000,000 instead of 1.0).
//! 
//! The Solution: We maintain a rolling Mean and Standard Deviation (Sigma).
//! Any gradient > 3.5 Sigma is rejected.

use serde::{Serialize, Deserialize};

/// A snapshot of the model's current gradient state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientSnapshot {
    pub mean: f64,
    pub variance: f64,
    pub count: u64,
    pub min: f64,
    pub max: f64,
}

/// The Security Engine.
pub struct AnomalyDetector {
    /// Key: Layer ID
    /// Value: Stats for that layer
    layers: std::collections::HashMap<u32, LayerStats>,
    /// Z-Score threshold. 3.0 is standard, 5.0 is loose.
    threshold: f64,
}

#[derive(Debug, Clone)]
struct LayerStats {
    mean: f64,
    m2: f64, // Sum of squares of differences from the current mean (Welford's algo)
    count: u64,
}

impl LayerStats {
    fn new() -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
        }
    }

    /// Welford's Online Algorithm:
    /// Allows us to update variance with O(1) memory.
    fn update(&mut self, new_value: f64) -> GradientSnapshot {
        self.count += 1;
        let delta = new_value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = new_value - self.mean;
        self.m2 += delta * delta2;

        let variance = if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        };

        GradientSnapshot {
            mean: self.mean,
            variance,
            count: self.count,
            min: self.mean - variance.sqrt(), // Simplified approximation
            max: self.mean + variance.sqrt(),
        }
    }
}

impl AnomalyDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            layers: std::collections::HashMap::new(),
            threshold,
        }
    }

    /// Checks a batch of gradients against the history.
    /// Returns `Ok` if safe, `Err` if suspicious.
    pub fn validate(&mut self, layer_id: u32, gradients: &[f32]) -> Result<(), String> {
        // 1. Calculate average magnitude of this incoming batch
        let incoming_avg: f64 = gradients.iter().map(|&x| x.abs() as f64).sum::<f64>() / gradients.len() as f64;

        // 2. Get or create stats for this layer
        let stats = self.layers.entry(layer_id).or_insert_with(LayerStats::new);

        // 3. Warm-up phase: If we haven't seen enough data, we blindly accept (Training Phase)
        if stats.count < 100 {
            stats.update(incoming_avg);
            tracing::debug!("Layer {}: Warmup phase. Accepting. Avg={}", layer_id, incoming_avg);
            return Ok(());
        }

        // 4. Anomaly Check
        let current_snapshot = stats.update(incoming_avg);
        let std_dev = current_snapshot.variance.sqrt();

        // Avoid divide by zero
        if std_dev < 1e-6 {
            return Ok(());
        }

        // Calculate Z-Score: (Incoming - Mean) / StdDev
        let z_score = (incoming_avg - current_snapshot.mean) / std_dev;

        if z_score.abs() > self.threshold {
            return Err(format!(
                "POISON DETECTED: Layer {} | Z-Score {:.2} > {:.2} | Mean {:.4}, Incoming {:.4}",
                layer_id, z_score, self.threshold, current_snapshot.mean, incoming_avg
            ));
        }

        Ok(())
    }
}