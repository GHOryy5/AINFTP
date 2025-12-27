//! ainftp-ebpf: Aggregation Logic
//! 
//! "In-Network Aggregation"
//! We sum up gradients from multiple nodes right here in the kernel.
//! This reduces traffic to userspace by a factor of 32 (BATCH_SIZE).

use aya_ebpf::maps::HashMap;
use core::mem;
use crate::maps::{LayerAccumulator, GradientPacket, BATCH_SIZE, MAX_PAYLOAD_SIZE, EVENTS, AGGREGATION_CACHE};

/// Processes a packet. 
/// Returns true if the packet was consumed (handled internally).
/// Returns false if the packet should be passed to userspace (or dropped).
pub fn handle_gradient(node_id: u32, layer_id: u32, chunk_idx: u32, data_ptr: usize, data_end: usize) -> bool {
    let key = (layer_id, chunk_idx);
    
    unsafe {
        let agg = AGGREGATION_CACHE.get_mut(&key);
        
        match agg {
            Some(acc) => {
                // Straggler Logic: If we already finished this batch, drop the late packet.
                if acc.count >= BATCH_SIZE {
                    return false; // Drop it
                }

                // AGGREGATION LOOP
                // Pointer arithmetic to read i16s from the packet
                let src = data_ptr as *const i16;
                let limit = data_end - data_ptr;
                
                for i in 0..MAX_PAYLOAD_SIZE {
                    if (i * 2) >= limit { break; } // Don't read past packet end
                    let val = *src.add(i);
                    acc.sum_i16[i] += val as i32;
                }
                
                acc.count += 1;

                // Batch Complete? Flush to userspace.
                if acc.count == BATCH_SIZE {
                    flush_to_userspace(acc);
                    AGGREGATION_CACHE.remove(&key);
                }
            }
            None => {
                // First packet of the batch. Create accumulator.
                let mut new_acc = LayerAccumulator {
                    layer_id,
                    chunk_idx,
                    count: 1,
                    sum_i16: [0; MAX_PAYLOAD_SIZE],
                };

                // Copy data for the first entry
                let src = data_ptr as *const i16;
                let limit = data_end - data_ptr;
                for i in 0..MAX_PAYLOAD_SIZE {
                    if (i * 2) >= limit { break; }
                    new_acc.sum_i16[i] = *src.add(i) as i32;
                }

                AGGREGATION_CACHE.insert(&key, &new_acc, 0).ok();
            }
        }
    }
    true // Packet consumed
}

/// Takes the sum, divides by count (averaging), and sends to userspace RingBuffer.
unsafe fn flush_to_userspace(acc: &LayerAccumulator) {
    // Reserve space in the ring buffer
    let reserve = EVENTS.reserve(0, mem::size_of::<GradientPacket>());
    
    if let Ok(mut slot) = reserve {
        let ptr = slot.as_mut_ptr() as *mut GradientPacket;
        
        // Construct the averaged packet
        (*ptr).layer_id = acc.layer_id;
        (*ptr).chunk_idx = acc.chunk_idx;
        
        // Average: Sum / Count (Integer division)
        for i in 0..MAX_PAYLOAD_SIZE {
            (*ptr).data[i] = (acc.sum_i16[i] / acc.count as i32) as i16;
        }
        
        slot.submit(0);
    }
}