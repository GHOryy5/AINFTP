//! ainftp Compute Layer
//! 
//! Everything related to GPU, RDMA, and low-level memory management.

pub mod rdma;
pub mod cuda; // We'll do this next

// Re-export for convenience
pub use rdma::{DmaBufferPool, GpuDirectRegion};