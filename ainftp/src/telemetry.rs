//! ainftp Telemetry
//! 
//! Exposes metrics for Prometheus or Grafana.
//! We want to see everything: GPU temp, Network jitter, Packet drops.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone)]
pub struct ClusterMetrics {
    pub start_time: Instant,
    
    // Counters
    pub packets_rx: Arc<AtomicU64>,
    pub packets_tx: Arc<AtomicU64>,
    pub stragglers_dropped: Arc<AtomicU64>,
    
    // Latency
    pub min_latency_us: Arc<AtomicU64>,
    pub max_latency_us: Arc<AtomicU64>,
}

impl ClusterMetrics {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            packets_rx: Arc::new(AtomicU64::new(0)),
            packets_tx: Arc::new(AtomicU64::new(0)),
            stragglers_dropped: Arc::new(AtomicU64::new(0)),
            min_latency_us: Arc::new(AtomicU64::new(u64::MAX)),
            max_latency_us: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn record_latency(&self, latency_us: u64) {
        // Update Max
        let mut current_max = self.max_latency_us.load(Ordering::Relaxed);
        loop {
            if latency_us <= current_max { break; }
            match self.max_latency_us.compare_exchange_weak(
                current_max, 
                latency_us, 
                Ordering::Relaxed, 
                Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }

        // Update Min
        let mut current_min = self.min_latency_us.load(Ordering::Relaxed);
        loop {
            if latency_us >= current_min { break; }
            match self.min_latency_us.compare_exchange_weak(
                current_min, 
                latency_us, 
                Ordering::Relaxed, 
                Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => current_min = x,
            }
        }
    }

    /// Returns a human-readable summary of the cluster health.
    pub fn summary(&self) -> String {
        let rx = self.packets_rx.load(Ordering::Relaxed);
        let dropped = self.stragglers_dropped.load(Ordering::Relaxed);
        let uptime = self.start_time.elapsed().as_secs();
        
        format!(
            "Uptime: {}s | RX: {} pkts | Dropped: {} | MinLat: {}us | MaxLat: {}us",
            uptime,
            rx,
            dropped,
            self.min_latency_us.load(Ordering::Relaxed),
            self.max_latency_us.load(Ordering::Relaxed)
        )
    }
}