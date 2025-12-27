//! ainftp-ebpf: Packet Parser
//! 
//! Direct memory access. If we mess up a boundary check, the kernel verifier kills us.
//! No mercy for unsafe code.

use aya_ebpf::{programs::XdpContext, ctty};
use core::mem;
use network_types::{
    eth::{EthHdr, EtherType},
    ip::{Ipv4Hdr, IpProto},
    udp::UdpHdr},
};

// Our custom protocol header (sits inside UDP payload)
#[repr(C, packed)]
pub struct AinHeader {
    pub magic: u32,       // 0x41494E50
    pub node_id: u32,
    pub layer_id: u32,
    pub chunk_idx: u32,
    pub timestamp: u64,
}

pub const AIN_MAGIC: u32 = 0x41494E50;

/// Extracts the AIN Header from the raw packet.
/// Returns None if the packet is malformed or not ours.
#[inline(always)]
pub fn parse_packet(ctx: &XdpContext) -> Option<(AinHeader, usize)> {
    let start = ctx.data();
    let end = ctx.data_end();

    // 1. Ethernet Check
    let eth_hlen = mem::size_of::<EthHdr>();
    if start.add(eth_hlen) > end { return None; }
    let eth = unsafe { &*(start as *const EthHdr) };
    if eth.ether_type != EtherType::Ipv4 { return None; }

    // 2. IP Check
    let ip_start = start.add(eth_hlen);
    let ip_hlen = mem::size_of::<Ipv4Hdr>();
    if ip_start.add(ip_hlen) > end { return None; }
    let ip = unsafe { &*(ip_start as *const Ipv4Hdr) };
    if ip.proto != IpProto::Udp { return None; }

    // 3. UDP Check
    let udp_start = ip_start.add(ip_hlen);
    let udp_hlen = mem::size_of::<UdpHdr>();
    if udp_start.add(udp_hlen) > end { return None; }

    // 4. Our Protocol Check
    let payload_start = udp_start.add(udp_hlen);
    let ain_hlen = mem::size_of::<AinHeader>();
    if payload_start.add(ain_hlen) > end { return None; }

    let ain = unsafe { &*(payload_start as *const AinHeader) };
    if ain.magic != AIN_MAGIC { return None; }

    // Return header and pointer to start of gradient data
    Some((*ain, payload_start.add(ain_hlen) as usize))
}

/// Helper to increment metrics safely
#[inline(always)]
pub fn inc_metric(field: MetricField) {
    unsafe {
        let stats = super::maps::STATS.get_ptr_mut(0);
        if !stats.is_null() {
            match field {
                MetricField::Packets => (*stats).packets_processed += 1,
                MetricField::Stragglers => (*stats).stragglers_dropped += 1,
                MetricField::Saved => (*stats).bytes_saved += 1, // simplified for now
            }
        }
    }
}

pub enum MetricField { Packets, Stragglers, Saved }