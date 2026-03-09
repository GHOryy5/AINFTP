use cudarc::driver::{CudaDevice, CudaFunction, LaunchConfig, CudaStream};
use anyhow::Result;
use std::sync::Arc;

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
    device: Arc<CudaDevice>,
    stream: CudaStream,
    sgd_kernel: CudaFunction,
}

impl GpuEngine {
    pub fn new(gpu_id: u32) -> Result<Self> {
        let dev = Arc::new(CudaDevice::new(gpu_id as usize)?);
        let kernel = unsafe { dev.load_ptx(SGD_KERNEL_PTX, "sgd_step", &["sgd_step"])? };
        let stream = dev.fork_stream()?;

        Ok(Self {
            device: dev,
            stream,
            sgd_kernel: kernel,
        })
    }

    pub fn register_dma_region(&self, dma_ptr: *mut std::ffi::c_void, len: usize) -> Result<()> {
        // CU_MEMHOSTREGISTER_DEVICEMAP = 0x02
        // CU_MEMHOSTREGISTER_PORTABLE = 0x01
        unsafe {
            let res = cudarc::driver::sys::cuMemHostRegister(dma_ptr, len, 0x02);
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                anyhow::bail!("cuMemHostRegister failed: {:?}", res);
            }
        }
        Ok(())
    }

    pub async fn launch_sgd(
        &self, 
        weights_gpu_ptr: u64,
        grads_dma_ptr: u64,
        lr: f32,
        n_elements: usize,
    ) -> Result<()> {
        let block_dim = 256;
        let grid_dim = (n_elements as u32 + block_dim - 1) / block_dim;

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.sgd_kernel.launch_on_stream(
                &self.stream, 
                cfg, 
                (weights_gpu_ptr, grads_dma_ptr, lr, n_elements as u32)
            )?;
        }

        Ok(())
    }
}

impl Clone for GpuEngine {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            stream: self.device.fork_stream().unwrap(),
            sgd_kernel: self.sgd_kernel.clone(),
        }
    }
}
