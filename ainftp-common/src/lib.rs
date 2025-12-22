#![no_std]

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NodeStats {
    pub total_packets: u64,
    pub total_bytes: u64,
    pub last_seen_ts: u64,
}

#[cfg(feature = "user")]
unsafe impl aya::Pod for NodeStats {}
