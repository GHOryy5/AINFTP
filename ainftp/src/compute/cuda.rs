//! CUDA Compute Engine - GPUDirect Zero-Copy Implementation
use cudarc::driver::{sys, CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use std::sync::Arc;
use anyhow::{Result, Context};
use tracing::{info, debug};

// PTX Kernel: w = w - (g * lr)
const SGD_KERNEL_PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry sgd_step(
    .param .u64 weights_ptr,
    .param .u64 grads_ptr,
    .param .f32 lr,
    .param .u32 n_elements
) {
    .reg .pred %p;
    .reg .f32 %f<4>;
    .reg .u64 %rd<4>;
    .reg .u32 %r<4>;

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mad.lo.u32 %r4, %r3, %r2, %r1;
    
    ld.param.u32 %r1, [n_elements];
    setp.ge.u32 %p1, %r4, %r1;
    @%p1 bra L_finish;

    ld.param.u64 %rd1, [weights_ptr];
    ld.param.u64 %rd2, [grads_ptr];
    ld.param.f32 %f1, [lr];

    mad.wide.u32 %rd3, %r4, 4, %rd2; 
    mad.wide.u32 %rd4, %r4, 4, %rd1; 

    ld.global.f32 %f2, [%rd3]; 
    ld.global.f32 %f3, [%rd4]; 

    mul.f32 %f2, %f2, %f1;   
    sub.f32 %f3, %f3, %f2;   

    st.global.f32 [%rd4], %f3;

L_finish:
    ret;
}
"#;

pub struct GpuEngine {
    pub device: Arc<CudaDevice>,
    sgd_kernel: CudaFunction,
}

impl GpuEngine {
    pub async fn new(gpu_id: i32) -> Result<Self> {
        // ordinal parameter is i32 in cudarc
        let dev = CudaDevice::new(gpu_id as usize)?; 
        info!(">> Initialized CUDA Device: {:?}", dev.name());

        let kernel = dev.load_ptx(SGD_KERNEL_PTX.into(), "sgd_step", &["sgd_step"])?;

        Ok(Self {
            device: dev.clone(),
            sgd_kernel: kernel,
        })
    }

    /// True Zero opy GPUDirect Magic.
    /// Locks CPU RAM (DMA buffer) and maps it to the GPU address space.
    /// Returns a CUdeviceptr that the GPU can natively read over the PCIe bus.
    pub fn pin_and_map_dma(&self, host_ptr: *mut std::ffi::c_void, len_bytes: usize) -> Result<sys::CUdeviceptr> {
        let mut d_ptr: sys::CUdeviceptr = 0;
        
        unsafe {
            // 1. Page lock the host memory & tell NVIDIA we will map it to the device
            let res = sys::cuMemHostRegister(
                host_ptr,
                len_bytes,
                sys::CU_MEMHOSTREGISTER_DEVICEMAP,
            );
            if res != sys::CUresult::CUDA_SUCCESS {
                anyhow::bail!("cuMemHostRegister failed: {:?}", res);
            }

            // 2. Extract the GPU-addressable pointer for the CPU memory
            let res = sys::cuMemHostGetDevicePointer(
                &mut d_ptr as *mut _,
                host_ptr,
                0,
            );
            if res != sys::CUresult::CUDA_SUCCESS {
                anyhow::bail!("cuMemHostGetDevicePointer failed: {:?}", res);
            }
        }

        debug!("Mapped CPU DMA {:p} -> GPU Ptr {:#x}", host_ptr, d_ptr);
        Ok(d_ptr)
    }

    /// Launches the SGD step.
    /// weights_d_ptr: The actual GPU VRAM pointer to the model weights.
    /// grads_mapped_ptr: The pointer returned by `pin_and_map_dma`.
    pub async fn launch_sgd(
        &self, 
        weights_d_ptr: sys::CUdeviceptr, 
        grads_mapped_ptr: sys::CUdeviceptr, 
        lr: f32,
        n_elements: u32,
    ) -> Result<()> {
        
        let block_dim = 256;
        let grid_dim = (n_elements + block_dim - 1) / block_dim;

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch directly using the device pointers. No HtoD copies.
        unsafe {
            self.sgd_kernel.clone().launch(
                cfg, 
                (weights_d_ptr, grads_mapped_ptr, lr, n_elements)
            ).context("CUDA PTX Kernel Launch Failed")?;
        }

        Ok(())
    }
}