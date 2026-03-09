
#![no_std]
#![no_main]

use aya_ebpf::{
    macros::{map, xdp},
    maps::{HashMap, PerCpuArray, RingBuf},
    programs::XdpContext,
    bindings::xdp_action,
};
use core::mem;
use network_types::{
    eth::{EthHdr, EtherType},
    ip::{Ipv4Hdr, IpProto},
    udp::UdpHdr,
};

const AIN_MAGIC: u32 = 0x41494E50; 
const MAX_PAYLOAD_SIZE: usize = 64; 
const BATCH_SIZE: u32 = 32;
const STRAGGLER_WINDOW_NS: u64 = 500_000_000;

#[repr(C, packed)]
pub struct AinHeader {
    pub magic: u32,
    pub node_id: u32,
    pub layer_id: u32,
    pub chunk_idx: u32,
    pub total_chunks: u32,
    pub timestamp: u64,
}

#[repr(C, packed)]
pub struct ShardID {
    pub layer: u32,
    pub chunk: u32,
}

#[repr(C)]
pub struct GradientPacket {
    pub header: AinHeader,
    pub data: [i16; MAX_PAYLOAD_SIZE],
}

#[repr(C)]
pub struct LayerAccumulator {
    pub layer_id: u32,
    pub chunk_idx: u32,
    pub count: u32,
    pub sum_i16: [i32; MAX_PAYLOAD_SIZE],
}

#[repr(C)]
pub struct CoordStats {
    pub mean_x1000: i32,
    pub m2_x1000: i64,
    pub count: u64,
}

#[map]
static mut ALLOWED_NODES: HashMap<u32, u32> = HashMap::with_max_entries(1024, 0);

#[map]
static mut STATE_HASHES: HashMap<ShardID, u32> = HashMap::with_max_entries(4096, 0);

#[map]
static mut AGGREGATION_CACHE: HashMap<(u32, u32), LayerAccumulator> = 
    HashMap::with_max_entries(4096, 0);

#[map]
static mut EVENTS: RingBuf = RingBuf::with_byte_size(1024 * 1024, 0);

#[map]
static mut STATS: PerCpuArray<Stats> = PerCpuArray::with_max_entries(1, 0);

#[map]
static mut NORMALCY_MAP: HashMap<u32, CoordStats> = HashMap::with_max_entries(MAX_PAYLOAD_SIZE as u32, 0);

#[map]
static mut CLUSTER_CLOCK: HashMap<u32, u64> = HashMap::with_max_entries(1, 0);

#[repr(C)]
pub struct Stats {
    pub packets_processed: u64,
    pub bytes_saved: u64,
    pub stragglers_dropped: u64,
    pub outliers_clipped: u64,
    pub state_errors_dropped: u64,
}

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
    
    if start + mem::size_of::<EthHdr>() > end { return Err(xdp_action::XDP_PASS); }
    let eth_hdr = unsafe { &*(start as *const EthHdr) };
    if eth_hdr.ether_type != EtherType::Ipv4 { return Ok(xdp_action::XDP_PASS); }

    let ip_start = start + mem::size_of::<EthHdr>();
    if ip_start + mem::size_of::<Ipv4Hdr>() > end { return Ok(xdp_action::XDP_PASS); }
    let ip_hdr = unsafe { &*(ip_start as *const Ipv4Hdr) };
    if ip_hdr.proto != IpProto::Udp { return Ok(xdp_action::XDP_PASS); }

    let udp_start = ip_start + mem::size_of::<Ipv4Hdr>();
    if udp_start + mem::size_of::<UdpHdr>() > end { return Ok(xdp_action::XDP_PASS); }

    let payload_start = udp_start + mem::size_of::<UdpHdr>();
    if payload_start + mem::size_of::<AinHeader>() > end { return Ok(xdp_action::XDP_PASS); }
    let ain_hdr = unsafe { &*(payload_start as *const AinHeader) };
    
    if ain_hdr.magic != AIN_MAGIC { return Ok(xdp_action::XDP_PASS); }

    unsafe {
        if ALLOWED_NODES.get(&ain_hdr.node_id).is_none() {
            return Ok(xdp_action::XDP_DROP);
        }
    }

    unsafe {
        let clock_key = 0u32;
        if let Some(&max_ts) = CLUSTER_CLOCK.get(&clock_key) {
            if ain_hdr.timestamp < max_ts.saturating_sub(STRAGGLER_WINDOW_NS) {
                inc_stats_straggler();
                return Ok(xdp_action::XDP_DROP);
            }
            if ain_hdr.timestamp > max_ts {
                let _ = CLUSTER_CLOCK.insert(&clock_key, &ain_hdr.timestamp, 0);
            }
        } else {
            let _ = CLUSTER_CLOCK.insert(&clock_key, &ain_hdr.timestamp, 0);
        }
    }

    let shard_id = ShardID { layer: ain_hdr.layer_id, chunk: ain_hdr.chunk_idx };
    unsafe {
        if let Some(&expected_hash) = STATE_HASHES.get(&shard_id) {
            let data_ptr = (payload_start + mem::size_of::<AinHeader>()) as *const u8;
            let actual_hash = compute_hash(data_ptr, end as *const u8);
            if actual_hash != expected_hash {
                inc_stats_state_error();
                return Ok(xdp_action::XDP_DROP);
            }
        }
    }

    let key = (ain_hdr.layer_id, ain_hdr.chunk_idx);
    let data_ptr = (payload_start + mem::size_of::<AinHeader>()) as *const i16;

    unsafe {
        let agg = AGGREGATION_CACHE.get_mut(&key);
        match agg {
            Some(acc) => {
                if acc.count >= BATCH_SIZE {
                    inc_stats_straggler();
                    return Ok(xdp_action::XDP_DROP);
                }

                for i in 0..MAX_PAYLOAD_SIZE {
                    if (data_ptr.add(i + 1) as usize) > end { break; }
                    let mut val = *data_ptr.add(i);
                    let val_x1000 = (val as i32) * 1000;

                    // --- BYZANTINE SHIELD: SELF-LEARNING WELFORD'S ---
                    if let Some(stats) = NORMALCY_MAP.get_mut(&(i as u32)) {
                        stats.count += 1;
                        let delta = val_x1000 - stats.mean_x1000;
                        stats.mean_x1000 += delta / stats.count as i32;
                        let delta2 = val_x1000 - stats.mean_x1000;
                        stats.m2_x1000 += (delta as i64) * (delta2 as i64);

                        let variance = if stats.count > 1 { stats.m2_x1000 / (stats.count as i64 - 1) } else { 0 };
                        // Simplified stddev check for kernel performance
                        if stats.count > 100 && (delta.abs() as i64) > (variance * 9) {
                            let sign = if val >= 0 { 1 } else { -1 };
                            val = (stats.mean_x1000 / 1000) as i16; // Reset to mean on spike
                            inc_stats_clipped();
                        }
                    } else {
                        let new_stats = CoordStats { mean_x1000: val_x1000, m2_x1000: 0, count: 1 };
                        let _ = NORMALCY_MAP.insert(&(i as u32), &new_stats, 0);
                    }

                    acc.sum_i16[i] += val as i32;
                }
                acc.count += 1;

                if acc.count == BATCH_SIZE {
                    flush_accumulator(acc);
                    AGGREGATION_CACHE.remove(&key);
                    inc_stats_saved((MAX_PAYLOAD_SIZE * (BATCH_SIZE as usize - 1)) as u64);
                }
            }
            None => {
                let mut new_acc = LayerAccumulator {
                    layer_id: ain_hdr.layer_id,
                    chunk_idx: ain_hdr.chunk_idx,
                    count: 1,
                    sum_i16: [0; MAX_PAYLOAD_SIZE],
                };
                for i in 0..MAX_PAYLOAD_SIZE {
                    if (data_ptr.add(i + 1) as usize) > end { break; }
                    new_acc.sum_i16[i] = *data_ptr.add(i) as i32;
                }
                let _ = AGGREGATION_CACHE.insert(&key, &new_acc, 0);
            }
        }
    }

    inc_stats_packets();
    Ok(xdp_action::XDP_DROP)
}

#[inline(always)]
unsafe fn flush_accumulator(agg: &LayerAccumulator) {
    if let Ok(mut slot) = EVENTS.reserve(0, mem::size_of::<GradientPacket>()) {
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

#[inline(always)]
unsafe fn compute_hash(mut ptr: *const u8, end: *const u8) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for _ in 0..128 {
        if ptr.add(4) > end { break; }
        h ^= *(ptr as *const u32);
        h = h.rotate_left(5);
        h = h.wrapping_add(0x27d4eb2d);
        ptr = ptr.add(4);
    }
    h
}

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
fn inc_stats_clipped() {
    unsafe {
        let s = STATS.get_ptr_mut(0);
        if !s.is_null() { (*s).outliers_clipped += 1; }
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
