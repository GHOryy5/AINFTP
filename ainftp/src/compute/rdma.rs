use std::ptr::NonNull;
use libc::{c_void, mmap, munmap, MAP_ANONYMOUS, MAP_PRIVATE, MAP_HUGETLB, PROT_READ, PROT_WRITE, MAP_FAILED};
use anyhow::{Result, bail};
use tracing::{info, warn};

/// Represents a chunk of memory that is "Pinned" in RAM.
/// The Pool owns this memory. Do not unmap it manually.
#[derive(Debug)]
pub struct GpuDirectRegion {
    ptr: NonNull<c_void>,
    size: usize,
}

impl GpuDirectRegion {
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub unsafe fn as_slice_mut<T>(&self, count: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, count)
    }
}

/// A pool of pre-allocated HugeTLB pages.
pub struct DmaBufferPool {
    free_list: Vec<GpuDirectRegion>,
    // We keep a master record so we can munmap everything on shutdown
    master_record: Vec<GpuDirectRegion>, 
    region_size: usize,
}

impl DmaBufferPool {
    pub fn new(count: usize, size: usize) -> Result<Self> {
        let mut free_list = Vec::with_capacity(count);
        let mut master_record = Vec::with_capacity(count);
        
        info!(">> Allocating {} MB of DMA memory (HugeTLB)...", (count * size) / 1_048_576);

        for _ in 0..count {
            let mut ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    size,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                    -1,
                    0,
                )
            };

            if ptr == MAP_FAILED {
                warn!("HugeTLB allocation failed. Falling back to standard mmap.");
                ptr = unsafe {
                    mmap(
                        std::ptr::null_mut(),
                        size,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS,
                        -1,
                        0,
                    )
                };
                
                if ptr == MAP_FAILED {
                    bail!("OOM: Failed to allocate DMA buffer pool.");
                }
            }

            let valid_ptr = NonNull::new(ptr).expect("mmap returned null instead of MAP_FAILED");
            
            let region = GpuDirectRegion { ptr: valid_ptr, size };
            
            // Clone the pointers (safe because we manage the lifecycle here)
            free_list.push(GpuDirectRegion { ptr: valid_ptr, size });
            master_record.push(region);
        }

        Ok(Self { free_list, master_record, region_size: size })
    }

    pub fn alloc(&mut self) -> Result<GpuDirectRegion> {
        self.free_list.pop().ok_or_else(|| anyhow::anyhow!("DMA Buffer Pool exhausted!"))
    }

    pub fn free(&mut self, region: GpuDirectRegion) {
        self.free_list.push(region);
    }
}

// Only unmap the memory when the ENTIRE pool shuts down.
impl Drop for DmaBufferPool {
    fn drop(&mut self) {
        info!(">> Tearing down DMA Pool. Unmapping memory...");
        for region in &self.master_record {
            unsafe {
                munmap(region.ptr.as_ptr(), region.size);
            }
        }
    }
}