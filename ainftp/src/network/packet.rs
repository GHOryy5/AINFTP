//! ainftp Protocol Specification
//! 
//! Defines the binary format of packets sent over the wire.
//! These MUST match the `#[repr(C)]` structs in `ainftp-ebpf`.

use serde::{Serialize, Deserialize};

/// Magic number to identify our traffic: 0x41494E50
pub const AIN_MAGIC: u32 = 0x41494E50;

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AinHeader {
    pub magic: u32,
    pub node_id: u32,
    pub layer_id: u32,
    pub chunk_idx: u32,
    pub total_chunks: u32,
    pub timestamp: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PacketType {
    Handshake = 0x01,
    Heartbeat = 0x02,
    Gradient = 0x10,
    StragglerWarning = 0xFF,
}

/// The wrapper for any message in our protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPacket {
    pub header: AinHeader,
    pub ptype: PacketType,
    // In a real scenario, `payload` would be a `Vec<u8>`.
    // For Gradient packets, we handle the buffer separately to avoid copying.
    pub payload_len: u16, 
}

impl NetworkPacket {
    /// Creates a new header for a specific node
    pub fn new_header(node_id: u32, layer_id: u32) -> AinHeader {
        AinHeader {
            magic: AIN_MAGIC,
            node_id,
            layer_id,
            chunk_idx: 0,
            total_chunks: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
        }
    }

    /// Serialize header to bytes (for direct socket writing)
    pub fn to_bytes(&self) -> Vec<u8> {
        // We need to manually serialize the AinHeader to ensure C-compatibility
        // because serde might add padding.
        // For now, assume `bincode` handles it correctly with `#[repr(C)]`.
        // In prod, we'd write `unsafe { std::ptr::read(...) }` to a Vec<u8>.
        vec![] // Placeholder
    }
}