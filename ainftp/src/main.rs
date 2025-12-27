use aya::maps::HashMap;
use aya::{include_bytes_aligned, Bpf};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use ainftp_common::{NodeStats, GradientChunk};
use std::{thread, time, sync::Arc};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // 1. Initialize the GPU (CUDA)
    let dev = CudaDevice::new(0)?; 
    println!("cuda: {}", dev.name()?);

    // 2. Load the eBPF "Synapse"
    let mut bpf = Bpf::load(include_bytes_aligned!(
        "../../target/x86_64-unknown-linux-gnu/release/ainftp"
    ))?;
    
    // Attach to the network interface
    let program: &mut aya::programs::Xdp = bpf.program_mut("ainftp").unwrap().try_into()?;
    program.load()?;
    program.attach("lo", aya::programs::XdpFlags::default())?;


    let mut grad_map: HashMap<_, u32, GradientChunk> = HashMap::try_from(bpf.map_mut("GRADIENT_CACHE").unwrap())?;

    loop {
        // 3. Poll the Kernel for new "Thoughts" (Gradients)
        for item in grad_map.iter() {
            let (layer_id, chunk) = item?;
            
            println!("New Gradient for Layer {} detected in Kernel!", layer_id);

            // 4. THE MAGIC: Move data from Kernel -> GPU Memory
            // In a real 'Frontier' setup, we'd use DMABUF for Zero-Copy, 
            // but here we push it to a CUDA buffer.
            let gpu_data = dev.htod_copy(chunk.weight_data.to_vec())?;
            
            println!("   -> Successfully injected into GPU memory.");
            
            // 5. Trigger a "Model Update" on the GPU
            // (Here you would launch a CUDA kernel to apply the gradients)
        }
        
        thread::sleep(time::Duration::from_millis(90));
    }
}
