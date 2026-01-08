//! ainftp-kernel: The Network Reflex (Ring -3)
//!
//! standard tcp/ip is lag. we intercept at the driver level (XDP).
//! this code handles gradient aggregation directly in the NIC to save CPU cycles.
//! we use a custom protocol 'AINT' to distinguish AI traffic from web trash.
//!
//! no floating point in kernel (eBPF restriction), so we quantize gradients to i16
//! and aggregate them using integer math. precision loss is acceptable for the speedup.

#![no_std]
#![no_main]

use aya_ebpf::{
    macros::{map, xdp, program},
    maps::{HashMap, PerCpuArray, RingBuf},
    programs::XdpContext,
    bindings::xdp_action,
    ctty,
};
use core::mem;
use network_types::{
    eth::{EthHdr, EtherType},
    ip::{Ipv4Hdr, IpProto},
    udp::UdpHdr,
};

// --- PROTOCOL DEFINITIONS ---
/// Magic bytes to identify our AI traffic. 0x41494E = "AIN"
const AIN_MAGIC: u32 = 0x41494E50; 
const MAX_PAYLOAD_SIZE: usize = 64; 

#[repr(C, packed)]
pub struct AinHeader {
    magic: u32,
    node_id: u32,
    layer_id: u32,
    chunk_idx: u32,
    total_chunks: u32,
    timestamp: u64,
}

#[repr(C)]
pub struct GradientPacket {
    header: AinHeader,
    // we store quantized gradients here (f32 -> i16)
    // saves 50% bandwidth and allows integer math
    data: [i16; MAX_PAYLOAD_SIZE], 
}

#[repr(C)]
pub struct LayerAccumulator {
    layer_id: u32,
    chunk_idx: u32,
    count: u32,       // how many nodes contributed?
    sum_i16: [i32; MAX_PAYLOAD_SIZE], // i32 to prevent overflow during sum
}

// --- MAPS (The "Reflex" Memory) ---

/// Tracks who is allowed to send training data.
/// prevents random internet noise from spiking our kernel. 
#[map]
static mut ALLOWED_NODES: HashMap<u32, u32> = HashMap::with_max_entries(1024, 0);

/// Stores the running sum of gradients per layer/chunk.
/// This allows us to do "In-Network Aggregation".
/// instead of sending 100 packets to userspace, we send 1 averaged packet.
#[map]
static mut AGGREGATION_CACHE: HashMap<(u32, u32), LayerAccumulator> = 
    HashMap::with_max_entries(4096, 0);

/// Ring buffer to blast processed data to userspace.
/// faster than hash map polling because it's event-driven.
#[map]
static mut EVENTS: RingBuf = RingBuf::with_byte_size(1024 * 1024, 0);

/// Per-CPU metrics. atomic counters are expensive, so we avoid them here.
#[map]
static mut STATS: PerCpuArray<Stats> = PerCpuArray::with_max_entries(1, 0);

#[repr(C)]
pub struct Stats {
    packets_processed: u64,
    bytes_saved: u64,
    stragglers_dropped: u64,
}

// --- SECURITY & CONFIG ---
/// Global config knobs
const BATCH_SIZE: u32 = 32; // Wait for 32 nodes before flushing

#[xdp]
pub fn ainftp(ctx: XdpContext) -> u32 {
    match try_handle_packet(ctx) {
        Ok(action) => action,
        Err(_) => xdp_action::XDP_PASS, // Fail open, don't kill the network
    }
}

#[inline(always)]
fn try_handle_packet(ctx: XdpContext) -> Result<u32, u32> {
    // 1. BOUNDARY CHECKS
    // if we mess this up, the kernel verifier kills us( this is the key ). be precise.
    let start = ctx.data();
    let end = ctx.data_end();
    
    let eth_hlen = mem::size_of::<EthHdr>();
    if start.add(eth_hlen) > end { return Err(xdp_action::XDP_PASS); }
    let eth_hdr = unsafe { &*(start as *const EthHdr) };
    
    // Filter: Only IPv4
    if eth_hdr.ether_type != EtherType::Ipv4 { 
        return Ok(xdp_action::XDP_PASS); 
    }

    let ip_start = start.add(eth_hlen);
    let ip_hlen = mem::size_of::<Ipv4Hdr>();
    if ip_start.add(ip_hlen) > end { return Err(xdp_action::XDP_PASS); }
    let ip_hdr = unsafe { &*(ip_start as *const Ipv4Hdr) };

    // Filter: Only UDP
    if ip_hdr.proto != IpProto::Udp { 
        return Ok(xdp_action::XDP_PASS); 
    }

    let udp_start = ip_start.add(ip_hlen);
    let udp_hlen = mem::size_of::<UdpHdr>();
    if udp_start.add(udp_hlen) > end { return Err(xdp_action::XDP_PASS); }
    
    // 2. PROTOCOL VALIDATION
    let payload_start = udp_start.add(udp_hlen);
    let ain_hlen = mem::size_of::<AinHeader>();
    
    // packet must have our header
    if payload_start.add(ain_hlen) > end { return Ok(xdp_action::XDP_PASS); }
    
    let ain_hdr = unsafe { &*(payload_start as *const AinHeader) };
    
    // verify magic number. if it's not ours, let linux handle it.
    if ain_hdr.magic != AIN_MAGIC {
        return Ok(xdp_action::XDP_PASS);
    }

    // 3. AUTHORIZATION CHECK
    // is this node in the whitelist?
    unsafe {
        if ALLOWED_NODES.get(&ain_hdr.node_id).is_none() {
            // unauthorized node trying to inject data? hard drop.
            return Ok(xdp_action::XDP_DROP);
        }
    }

    // 4. AGGREGATION LOGIC (The secret sauce)
    // we identify the chunk by (layer_id, chunk_idx)
    let key = (ain_hdr.layer_id, ain_hdr.chunk_idx);
    
    unsafe {
        let agg = AGGREGATION_CACHE.get_mut(&key);
        
        match agg {
            Some(accumulator) => {
                // Straggler Check: if we already closed this batch, drop late packets
                if accumulator.count >= BATCH_SIZE {
                    inc_stats_straggler();
                    return Ok(xdp_action::XDP_DROP);
                }

                // pointer to the payload data
                let data_ptr = payload_start.add(ain_hlen) as *const i16;
                
                // Aggregate: Sum the i16 values into the i32 accumulator
                // loop unrolling could go here for speed, but verifier hates complex loops
                for i in 0..MAX_PAYLOAD_SIZE {
                    if payload_start.add(ain_hlen + i * 2) > end { break; }
                    let val = *data_ptr.add(i);
                    accumulator.sum_i16[i] += val as i32;
                }
                
                accumulator.count += 1;

                // Check if batch is full
                if accumulator.count == BATCH_SIZE {
                    flush_accumulator(accumulator);
                    // reset for next round
                    AGGREGATION_CACHE.remove(&key);
                    inc_stats_saved((MAX_PAYLOAD_SIZE * (BATCH_SIZE as usize - 1)) as u64);
                }
            }
            None => {
                // New chunk detected. Create a fresh accumulator.
                let new_agg = LayerAccumulator {
                    layer_id: ain_hdr.layer_id,
                    chunk_idx: ain_hdr.chunk_idx,
                    count: 1,
                    sum_i16: [0; MAX_PAYLOAD_SIZE], // zero init
                };
                
                // we have to copy data manually for the first entry
                // since we can't init the array easily in one go
                let data_ptr = payload_start.add(ain_hlen) as *const i16;
                let mut temp_sum = [0i32; MAX_PAYLOAD_SIZE];
                for i in 0..MAX_PAYLOAD_SIZE {
                     if payload_start.add(ain_hlen + i * 2) > end { break; }
                     temp_sum[i] = *data_ptr.add(i) as i32;
                }

                // Note: bpf map insert is heavy. we do it once.
                // cloning logic is verbose because of no-std
                let mut entry = LayerAccumulator { sum_i16: temp_sum, ..new_agg };
                AGGREGATION_CACHE.insert(&key, &entry, 0).map_err(|_| xdp_action::XDP_PASS)?;
            }
        }
    }

    // 5. INCREMENT METRICS
    inc_stats_packets();
    
    // We handled the packet in-kernel. Userspace doesn't need to see the raw packet.
    // XDP_DROP here means we consumed it.
    Ok(xdp_action::XDP_DROP)
}

/// Flushes the aggregated result to the Userspace RingBuffer.
/// Userspace will receive 1 packet instead of 32.
#[inline(always)]
unsafe fn flush_accumulator(agg: &LayerAccumulator) {
    let reserve = EVENTS.reserve(0, mem::size_of::<GradientPacket>()); // flags = 0
    
    if let Ok(mut slot) = reserve {
        let ptr = slot.as_mut_ptr() as *mut GradientPacket;
        
        // construct the result packet
        (*ptr).header.magic = AIN_MAGIC;
        (*ptr).header.layer_id = agg.layer_id;
        (*ptr).header.chunk_idx = agg.chunk_idx;
        (*ptr).header.node_id = 0; // 0 means "Aggregated"
        
        // Convert sum back to average (integer division)
        // This is the "Mean Gradient" being sent to GPU
        for i in 0..MAX_PAYLOAD_SIZE {
            (*ptr).data[i] = (agg.sum_i16[i] / BATCH_SIZE as i32) as i16;
        }
        
        slot.submit(0);
    }
}

// --- METRICS HELPERS ---
/// increment packet counter
#[inline(always)]
fn inc_stats_packets() {
    unsafe {
        let s = STATS.get_ptr_mut(0);
        if !s.is_null() {
            (*s).packets_processed += 1;
        }
    }
}

/// increment straggler counter
#[inline(always)]
fn inc_stats_straggler() {
    unsafe {
        let s = STATS.get_ptr_mut(0);
        if !s.is_null() {
            (*s).stragglers_dropped += 1;
        }
    }
}

/// increment bytes saved counter
#[inline(always)]
fn inc_stats_saved(bytes: u64) {
    unsafe {
        let s = STATS.get_ptr_mut(0);
        if !s.is_null() {
            (*s).bytes_saved += bytes;
        }
    }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}