//! ainftp-ebpf: Packet Parser
//! 
//! Direct memory access. If we mess up a boundary check, the kernel verifier kills us.
//! No mercy for unsafe code.

use aya_ebpf::{programs::XdpContext, ctty};
use core::mem;
use core::ptr;
use network_types::{
    eth::{EthHdr, EtherType},
    ip::IpProto,
    udp::UdpHdr,
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

    // 2. IP Check — be robust: handle variable IHL and fragmentation
    let ip_start = start.add(eth_hlen);
    // minimal IPv4 header is 20 bytes
    let ip_min_hlen = 20usize;
    if ip_start.add(ip_min_hlen) > end { return None; }

    // read version/IHL byte and compute header length
    let ver_ihl = unsafe { * (ip_start as *const u8) };
    let ihl = ((ver_ihl & 0x0f) as usize) * 4;
    if ihl < ip_min_hlen { return None; }
    if ip_start.add(ihl) > end { return None; }

    // protocol is at offset 9
    let proto = unsafe { *ip_start.add(9) };
    if proto != IpProto::Udp as u8 { return None; }

    // check fragmentation: flags+frag_offset at bytes 6-7 (network order)
    let frag_field_be = unsafe { ptr::read_unaligned(ip_start.add(6) as *const u16) };
    let frag_field = u16::from_be(frag_field_be);
    let frag_offset = frag_field & 0x1fff;
    let more_fragments = (frag_field & 0x2000) != 0;
    if frag_offset != 0 || more_fragments { return None; }

    // 3. UDP Check
    let udp_start = ip_start.add(ihl);
    let udp_hlen = mem::size_of::<UdpHdr>();
    if udp_start.add(udp_hlen) > end { return None; }

    // validate UDP length field (bytes 4-5 of UDP header)
    let udp_len_be = unsafe { ptr::read_unaligned(udp_start.add(4) as *const u16) };
    let udp_len = u16::from_be(udp_len_be) as usize;
    if udp_len < udp_hlen { return None; }
    if udp_start.add(udp_len) > end { return None; }

    // 4. Our Protocol Check
    let payload_start = udp_start.add(udp_hlen);
    let ain_hlen = mem::size_of::<AinHeader>();
    if payload_start.add(ain_hlen) > end { return None; }

    // Use an unaligned read to avoid potential UB / verifier issues with packed structs
    let ain = unsafe { ptr::read_unaligned(payload_start as *const AinHeader) };
    if ain.magic != AIN_MAGIC { return None; }

    // Return header and offset (relative to packet start) to the gradient data
    Some((ain, payload_start.add(ain_hlen) as usize - start as usize))
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