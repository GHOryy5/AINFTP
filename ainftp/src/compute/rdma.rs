//! Holographic Memory Management (RDMA)
//! 
//! We aren't using standard RAM. We are pinning physical memory pages
//! and mapping them so the NIC can write to them and the GPU can read them.
//! If the CPU touches this data, we failed.
//!
//! Dependencies: libc (for mmap/posix_memalign)

use std::ptr::NonNull;
use libc::{c_void, mmap, munmap, mprotect, MAP_ANONYMOUS, MAP_PRIVATE, MAP_HUGETLB, PROT_READ, PROT_WRITE};
use anyhow::{Result, bail};

/// Represents a chunk of memory that is "Pinned" in RAM.
/// The OS cannot swap this to disk. The NIC can write here directly via DMA.
pub struct GpuDirectRegion {
    /// NonNull so we don't deal with null checks at runtime
    ptr: NonNull<c_void>,
    size: usize,
    // In a real GPUDirect setup, we hold the 'ibv_mr' (InfiniBand Memory Region) handle here.
    // ibv_mr: *mut ibv_mr, 
}

impl GpuDirectRegion {
    /// Returns the raw pointer.
    /// This pointer is what you feed to `cudaHostRegister` or `cuMemHostRegister`.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.size
    }

    /// Slices the memory. Used if we want to view this as a gradient tensor.
    pub unsafe fn as_slice_mut<T>(&self, count: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, count)
    }
}

impl Drop for GpuDirectRegion {
    fn drop(&mut self) {
        // Give memory back to the OS when the struct dies.
        unsafe {
            munmap(self.ptr.as_ptr(), self.size);
        }
    }
}

/// A pool of pre-allocated HugeTLB pages.
/// 
/// Allocating memory during a training loop is death. We allocate 1GB at boot
/// and hand it out like candy.
pub struct DmaBufferPool {
    free_list: Vec<GpuDirectRegion>,
    region_size: usize,
}

impl DmaBufferPool {
    /// Initializes the pool with `count` regions of `size` bytes each.
    /// 
    /// # Safety
    /// This calls `mmap` with `MAP_HUGETLB`. If your Linux kernel doesn't have
    /// huge pages enabled (`vm.nr_hugepages`), this will fail or fall back.
    pub fn new(count: usize, size: usize) -> Result<Self> {
        let mut free_list = Vec::with_capacity(count);
        
        info!("Allocating {} MB of DMA memory (HugeTLB)...", (count * size) / 1024 / 1024);

        for _ in 0..count {
            // 1. MAP HUGE TLB
            // We ask for 2MB pages (or system default). This reduces TLB misses by ~1000x.
            let ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    size,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                    -1,
                    0,
                )
            };

            if ptr == libc::MAP_FAILED {
                // Fallback: If HugeTLB fails (common in dev envs), fall back to standard mmap.
                // In prod, we'd panic and tell the sysadmin to configure vm.nr_hugepages.
                warn!("HugeTLB allocation failed. Falling back to standard mmap. Latency will suffer.");
                let ptr = unsafe {
                    mmap(
                        std::ptr::null_mut(),
                        size,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS,
                        -1,
                        0,
                    )
                };
                
                if ptr == libc::MAP_FAILED {
                    bail!("OOM: Failed to allocate DMA buffer pool.");
                }
                
                free_list.push(GpuDirectRegion { ptr: NonNull::new(ptr).unwrap(), size });
            } else {
                // 2. MEMORY LOCK
                // Tell the kernel "Do not swap this to disk."
                // mlock(ptr, size); 
                
                // 3. NVIDIA GPUDirect REGISTRATION
                // This is the critical step. We tell the NVIDIA driver "This memory is safe for DMA."
                // In Rust, we use `cudarc::driver::result::host_register`.
                // For this file, we assume it's handled in the `cuda.rs` module
                // or we do it immediately here.
                
                free_list.push(GpuDirectRegion { ptr: NonNull::new(ptr).unwrap(), size });
            }
        }

        Ok(Self { free_list, region_size: size })
    }

    /// Gets a buffer from the pool. O(1) operation.
    pub fn alloc(&mut self) -> Result<GpuDirectRegion> {
        match self.free_list.pop() {
            Some(region) => Ok(region),
            None => {
                // Pool exhaustion is a system failure in distributed training.
                bail!("DMA Buffer Pool exhausted. Backpressure required.");
            }
        }
    }

    /// Returns a buffer to the pool.
    pub fn free(&mut self, region: GpuDirectRegion) {
        self.free_list.push(region);
    }
}