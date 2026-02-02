

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
    udp::UdpHdr},
};

// PROTOCOL DEFINITIONS 
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

// Locally defined ShardID (Mirrors ainftp-core)
#[repr(C, packed)]
#[derive(PartialEq)]
pub struct ShardID {
    pub layer: u32,
    pub chunk: u32,
}

#[repr(C)]
pub struct GradientPacket {
    header: AinHeader,
    data: [i16; MAX_PAYLOAD_SIZE], 
}

#[repr(C)]
pub struct LayerAccumulator {
    layer_id: u32,
    chunk_idx: u32,
    count: u32,
    sum_i16: [i32; MAX_PAYLOAD_SIZE], 
}

// MAPS (The "Reflex" Memory) 

/// Tracks who is allowed to send training data.
/// prevents random internet noise from spiking our kernel.
#[map]
static mut ALLOWED_NODES: HashMap<u32, u32> = HashMap::with_max_entries(1024, 0);

/// The NEW MAP: State Consensus.
/// Userspace maintains the "Golden Copy" of the model.
/// It pushes (ShardID -> Hash) here.
/// We verify incoming packets against this hash BEFORE aggregation.
#[map]
static mut STATE_HASHES: HashMap<ShardID, u32> = HashMap::with_max_entries(4096, 0);

/// Stores the running sum of gradients per layer/chunk.
/// This allows us to do "In-Network Aggregation".
#[map]
static mut AGGREGATION_CACHE: HashMap<(u32, u32), LayerAccumulator> = 
    HashMap::with_max_entries(4096, 0);

/// Ring buffer to blast processed data to userspace.
#[map]
static mut EVENTS: RingBuf = RingBuf::with_byte_size(1024 * 1024, 0);

/// Per-CPU metrics.
#[map]
static mut STATS: PerCpuArray<Stats> = PerCpuArray::with_max_entries(1, 0);

#[repr(C)]
pub struct Stats {
    packets_processed: u64,
    bytes_saved: u64,
    stragglers_dropped: u64,
    state_errors_dropped: u64, // New metric
}

// --- SECURITY & CONFIG ---
const BATCH_SIZE: u32 = 32; 

#[xdp]
pub fn ainftp(ctx: XdpContext) -> u32 {
    match try_handle_packet(ctx) {
        Ok(action) => action,
        Err(_) => xdp_action::XDP_PASS, 
    }
}

#[inline(always)]
fn try_handle_packet(ctx: XdpContext) -> Result<u32, u32> {
    let start = ctx.data();
    let end = ctx.data_end();
    
    // 1. BOUNDARY CHECKS
    let eth_hlen = mem::size_of::<EthHdr>();
    if start.add(eth_hlen) > end { return Err(xdp_action::XDP_PASS); }
    let eth_hdr = unsafe { &*(start as *const EthHdr) };
    
    if eth_hdr.ether_type != EtherType::Ipv4 { 
        return Ok(xdp_action::XDP_PASS); 
    }

    let ip_start = start.add(eth_hlen);
    let ip_hlen = mem::size_of::<Ipv4Hdr>();
    if ip_start.add(ip_hlen) > end { return Err(xdp_action::XDP_PASS); }
    let ip_hdr = unsafe { &*(ip_start as *const Ipv4Hdr) };

    if ip_hdr.proto != IpProto::Udp { 
        return Ok(xdp_action::XDP_PASS); 
    }

    let udp_start = ip_start.add(ip_hlen);
    let udp_hlen = mem::size_of::<UdpHdr>();
    if udp_start.add(udp_hlen) > end { return Err(xdp_action::XDP_PASS); }
    
    // 2. PROTOCOL VALIDATION
    let payload_start = udp_start.add(udp_hlen);
    let ain_hlen = mem::size_of::<AinHeader>();
    
    if payload_start.add(ain_hlen) > end { return Ok(xdp_action::XDP_PASS); }
    
    let ain_hdr = unsafe { &*(payload_start as *const AinHeader) };
    
    if ain_hdr.magic != AIN_MAGIC {
        return Ok(xdp_action::XDP_PASS);
    }

    // 3. AUTHORIZATION CHECK
    unsafe {
        if ALLOWED_NODES.get(&ain_hdr.node_id).is_none() {
            return Ok(xdp_action::XDP_DROP);
        }
    }

    // 4. STATE CONSENSUS CHECK (The "One of One" Move)
    // We check if this specific shard matches the Global State.
    let shard_id = ShardID { layer: ain_hdr.layer_id, chunk: ain_hdr.chunk_idx };
    
    unsafe {
        if let Some(&expected_hash) = STATE_HASHES.get(&shard_id) {
            // We have an expected hash. Verify it.
            let actual_hash = compute_hash(payload_start.add(ain_hlen), end);
            
            if actual_hash != expected_hash {
                // State mismatch. Desync or Poison.
                inc_stats_state_error();
                return Ok(xdp_action::XDP_DROP);
            }
        }
        // If no hash in map, we are in "Warmup" or new epoch. Accept.
    }

    // 5. AGGREGATION LOGIC
    let key = (ain_hdr.layer_id, ain_hdr.chunk_idx);
    
    unsafe {
        let agg = AGGREGATION_CACHE.get_mut(&key);
        
        match agg {
            Some(accumulator) => {
                if accumulator.count >= BATCH_SIZE {
                    inc_stats_straggler();
                    return Ok(xdp_action::XDP_DROP);
                }

                let data_ptr = payload_start.add(ain_hlen) as *const i16;
                
                for i in 0..MAX_PAYLOAD_SIZE {
                    if payload_start.add(ain_hlen + i * 2) > end { break; }
                    let val = *data_ptr.add(i);
                    accumulator.sum_i16[i] += val as i32;
                }
                
                accumulator.count += 1;

                if accumulator.count == BATCH_SIZE {
                    flush_accumulator(accumulator);
                    AGGREGATION_CACHE.remove(&key);
                    inc_stats_saved((MAX_PAYLOAD_SIZE * (BATCH_SIZE as usize - 1)) as u64);
                }
            }
            None => {
                let new_agg = LayerAccumulator {
                    layer_id: ain_hdr.layer_id,
                    chunk_idx: ain_hdr.chunk_idx,
                    count: 1,
                    sum_i16: [0; MAX_PAYLOAD_SIZE],
                };
                
                let data_ptr = payload_start.add(ain_hlen) as *const i16;
                let mut temp_sum = [0i32; MAX_PAYLOAD_SIZE];
                for i in 0..MAX_PAYLOAD_SIZE {
                     if payload_start.add(ain_hlen + i * 2) > end { break; }
                     temp_sum[i] = *data_ptr.add(i) as i32;
                }

                let mut entry = LayerAccumulator { sum_i16: temp_sum, ..new_agg };
                AGGREGATION_CACHE.insert(&key, &entry, 0).map_err(|_| xdp_action::XDP_PASS)?;
            }
        }
    }

    inc_stats_packets();
    Ok(xdp_action::XDP_DROP)
}

#[inline(always)]
unsafe fn flush_accumulator(agg: &LayerAccumulator) {
    let reserve = EVENTS.reserve(0, mem::size_of::<GradientPacket>()); 
    
    if let Ok(mut slot) = reserve {
        let ptr = slot.as_mut_ptr() as *mut GradientPacket;
        
        (*ptr).header.magic = AIN_MAGIC;
        (*ptr).header.layer_id = agg.layer_id;
        (*ptr).header.chunk_idx = agg.chunk_idx;
        (*ptr).header.node_id = 0;
        
        for i in 0..MAX_PAYLOAD_SIZE {
            (*ptr).data[i] = (agg.sum_i16[i] / BATCH_SIZE as i32) as i16;
        }
        
        slot.submit(0);
    }
}

// --- HASH HELPERS ---
/// Simple XOR-hash for kernel verification.
/// Fast, collision resistant enough for integrity checks.
#[inline(always)]
unsafe fn compute_hash(start: *const u8, end: *const u8) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    let mut ptr = start;
    
    while ptr < end {
        h ^= *(ptr as *const u32);
        h = h.rotate_left(5);
        h = h.wrapping_add(0x27d4eb2d);
        ptr = ptr.add(4);
    }
    h
}

// METRICS HELPERS 
#[inline(always)]
fn inc_stats_packets() {
    unsafe {
        let s = STATS.get_ptr_mut(0);
        if !s.is_null() { (*s).packets_processed += 1; }
    }
}

#[inline(always)]
fn inc_stats_straggler() {
    unsafe {
        let s = STATS.get_ptr_mut(0);
        if !s.is_null() { (*s).stragglers_dropped += 1; }
    }
}

#[inline(always)]
fn inc_stats_saved(bytes: u64) {
    unsafe {
        let s = STATS.get_ptr_mut(0);
        if !s.is_null() { (*s).bytes_saved += bytes; }
    }
}

#[inline(always)]
fn inc_stats_state_error() {
    unsafe {
        let s = STATS.get_ptr_mut(0);
        if !s.is_null() { (*s).state_errors_dropped += 1; }
    }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}