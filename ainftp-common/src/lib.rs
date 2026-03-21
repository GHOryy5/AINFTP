#![no_std]

// 64-byte alignment eliminates false sharing in L3 cache
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct NodeStats {
    pub total_packets: u64,
    pub total_bytes: u64,
    pub quantized_bytes: u64,
    pub welford_anomalies: u64,
    pub last_seen_ts: u64,
    pub _pad: [u64; 3],
}

// custom wire-protocol header
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AinftpHeader {
    pub magic: u16,
    pub version: u8,
    pub flags: u8,
    pub node_id: u32,
    pub sequence: u64,
}

// zero-copy ring buffer event to wake userspace
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GradientReadyEvent {
    pub node_id: u32,
    pub batch_id: u32,
    pub buffer_ptr: u64,
    pub checksum: u64,
}

#[cfg(feature = "user")]
unsafe impl aya::Pod for NodeStats {}

#[cfg(feature = "user")]
unsafe impl aya::Pod for AinftpHeader {}

#[cfg(feature = "user")]
unsafe impl aya::Pod for GradientReadyEvent {}