#![no_std]
#![no_main]

use aya_ebpf::{
    macros::{xdp, map},
    maps::HashMap,
    programs::XdpContext,
    bindings::xdp_action,
};
use core::mem;
use network_types::{
    eth::{EthHdr, EtherType},
    ip::{Ipv4Hdr, IpProto},
    udp::UdpHdr,
};

// Represents a chunk of a model's weights (gradients)
#[repr(C)]
pub struct GradientChunk {
    pub layer_id: u32,
    pub weight_data: [f32; 16], // Small 16-float chunk for demo
}

// A high-speed map to store incoming model updates from the "Distro"
#[map]
static mut GRADIENT_CACHE: HashMap<u32, GradientChunk> = HashMap::with_max_entries(4096, 0);

#[xdp]
pub fn ainftp(ctx: XdpContext) -> u32 {
    let _ = try_accumulate(ctx);
    xdp_action::XDP_PASS
}

fn try_accumulate(ctx: XdpContext) -> Result<u32, ()> {
    let start = ctx.data();
    let end = ctx.data_end();

    // 1. Standard Headers
    let eth_len = mem::size_of::<EthHdr>();
    if start + eth_len > end { return Ok(xdp_action::XDP_PASS); }
    let eth = unsafe { &*(start as *const EthHdr) };
    if eth.ether_type != EtherType::Ipv4 { return Ok(xdp_action::XDP_PASS); }

    let ip_start = start + eth_len;
    let ip_len = mem::size_of::<Ipv4Hdr>();
    if ip_start + ip_len > end { return Ok(xdp_action::XDP_PASS); }
    let ip = unsafe { &*(ip_start as *const Ipv4Hdr) };

    // 2. Filter for UDP (Used for high-speed AI sync like NCCL)
    if ip.proto != IpProto::Udp { return Ok(xdp_action::XDP_PASS); }
    
    let udp_start = ip_start + ip_len;
    let udp_len = mem::size_of::<UdpHdr>();
    if udp_start + udp_len > end { return Ok(xdp_action::XDP_PASS); }

    // 3. The "Deep Research" Part: Extract Gradient Data
    // We assume the payload starts right after the UDP header
    let payload_start = udp_start + udp_len;
    let payload_len = mem::size_of::<GradientChunk>();
    
    if payload_start + payload_len <= end {
        let chunk = unsafe { &*(payload_start as *const GradientChunk) };
        
        // Save to Kernel Map - The GPU will poll this map directly!
        unsafe {
            GRADIENT_CACHE.insert(&chunk.layer_id, chunk, 0).map_err(|_| ())?;
        }
    }

    Ok(xdp_action::XDP_PASS)
}
