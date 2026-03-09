use std::ptr::{NonNull, null_mut};
use libc::{c_void, mmap, munmap, mlock, MAP_ANONYMOUS, MAP_PRIVATE, MAP_HUGETLB, PROT_READ, PROT_WRITE};
use anyhow::{Result, bail};

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

impl Drop for GpuDirectRegion {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr.as_ptr(), self.size);
        }
    }
}

pub struct DmaBufferPool {
    free_list: Vec<GpuDirectRegion>,
    region_size: usize,
}

impl DmaBufferPool {
    pub fn new(count: usize, size: usize) -> Result<Self> {
        let mut free_list = Vec::with_capacity(count);
        
        for _ in 0..count {
            let ptr = unsafe {
                mmap(
                    null_mut(),
                    size,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                    -1,
                    0,
                )
            };

            let ptr = if ptr == libc::MAP_FAILED {
                unsafe {
                    mmap(
                        null_mut(),
                        size,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS,
                        -1,
                        0,
                    )
                }
            } else {
                ptr
            };

            if ptr == libc::MAP_FAILED {
                bail!("OOM: Failed to allocate DMA buffer pool.");
            }

            unsafe {
                if mlock(ptr, size) != 0 {
                    // Non-fatal but suboptimal
                }
            }

            free_list.push(GpuDirectRegion {
                ptr: NonNull::new(ptr).unwrap(),
                size
            });
        }

        Ok(Self { free_list, region_size: size })
    }

    pub fn alloc(&mut self) -> Result<GpuDirectRegion> {
        self.free_list.pop().ok_or_else(|| anyhow::anyhow!("Pool exhausted"))
    }

    pub fn free(&mut self, region: GpuDirectRegion) {
        self.free_list.push(region);
    }
}
