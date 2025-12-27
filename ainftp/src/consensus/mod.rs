//! ainftp Consensus Layer
//! 
//! Handles Byzantine fault tolerance and anomaly detection.
//! Prevents "Poison Gradients" from wrecking the model.

pub mod sentry;

pub use sentry::{AnomalyDetector, GradientSnapshot};