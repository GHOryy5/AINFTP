

mod compute;
mod network;
mod consensus;
mod telemetry;

use aya::{include_bytes_aligned, Bpf, programs::{Xdp, XdpFlags}};
use aya::maps::perf::AsyncPerfEventArray;
use aya::util::online_cpus;
use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio::{select, signal};
use tracing::{info, error, warn, debug};
use std::time::Duration;

// Imports
use compute::{rdma::DmaBufferPool, cuda::GpuEngine};
use network::{swarm::Swarm, packet::AinHeader}; 
use consensus::sentry::AnomalyDetector;
use telemetry::ClusterMetrics;

// CONSTANTS 
const GPU_ID: u32 = 0;
const DMA_POOL_SIZE: usize = 64;     
const DMA_REGION_SIZE: usize = 1 << 20; 
const SENTRY_THRESHOLD: f64 = 3.5;  

/// The Main Runtime
pub struct ReflexRuntime {
    // 1. Kernel Hook
    _bpf: Bpf,
    
    // 2. Subsystems
    _pool: DmaBufferPool,    // The "Holographic" Memory
    gpu: GpuEngine,         // The Compute Engine
    sentry: AnomalyDetector, // Security Guard
    _swarm_rx: mpsc::Receiver<network::swarm::SwarmEvent>, 
    
    // 4. Observability
    metrics: ClusterMetrics,
}

impl ReflexRuntime {
    /// Initialize all subsystems
    pub async fn init() -> Result<Self> {
        info!("--- Initializing ainftp v2.0 ---");

        // 1. Load eBPF Kernel Module
        let mut bpf = Bpf::load(include_bytes_aligned!(
            "../../target/bpfel-unknown-none/release/ainftp"
        ))?;
        let prog: &mut Xdp = bpf.program_mut("ainftp").unwrap().try_into()?;
        prog.load()?;
        prog.attach("eth0", XdpFlags::default())?; 
        info!(">> Kernel Reflex: ACTIVE");

        // 2. Start Network Swarm
        let (swarm, swarm_rx) = Swarm::bootstrap().await.context("Swarm failed")?;
        tokio::spawn(swarm.run(swarm_rx.clone())); 
        info!(">> Swarm: ACTIVE");

        // 3. Initialize Zero-Copy Memory Pool
        let pool = DmaBufferPool::new(DMA_POOL_SIZE, DMA_REGION_SIZE)?;
        info!(">> DMA Pool: {} MB Allocated", (DMA_POOL_SIZE * DMA_REGION_SIZE) / 1_048_576);

        // 4. Initialize GPU Engine
        let gpu = GpuEngine::new(GPU_ID).await?;
        info!(">> CUDA Engine: READY");

        // 5. Initialize Security Sentry
        let sentry = AnomalyDetector::new(SENTRY_THRESHOLD);
        info!(">> Security Sentry: MONITORING");

        let metrics = ClusterMetrics::new();

        Ok(Self {
            _bpf: bpf,
            _pool: pool,
            gpu,
            sentry,
            _swarm_rx: swarm_rx,
            metrics,
        })
    }

    /// The Core Loop
    pub async fn run(mut self) -> Result<()> {
        let mut perf_array = AsyncPerfEventArray::try_from(self._bpf.map_mut("EVENTS")?)?;
        let mut handlers = JoinSet::new();

        // --- WORKER POOL ---
        for cpu_id in online_cpus()? {
            let mut buf_reader = perf_array.open(cpu_id, None)?;
            let gpu = self.gpu.clone();
            let sentry = self.sentry.clone(); // Move sentry into worker
            
            handlers.spawn(async move {
                // Reusable buffer to avoid heap allocs
                let mut read_buf = vec![0u8; 1024 * 64]; // 64KB packets
                // Reusable buffer for f32 conversion
                let mut grads_f32 = vec![0f32; 1024 * 32]; // Max size
                
                loop {
                    select! {
                        // 1. READ FROM KERNEL
                        _ = buf_reader.read_events(&mut [read_buf.as_mut_slice()]) => {
                            if read_buf.is_empty() { continue }

                            // 2. PARSE HEADER
                            if read_buf.len() < std::mem::size_of::<AinHeader>() { continue } 
                            let header = unsafe { &*(read_buf.as_ptr() as *const AinHeader) };

                            debug!("RX Layer {} Chunk {}", header.layer_id, header.chunk_idx);

                            // 3. PARSE PAYLOAD (Zero-Copy Conversion)
                            let payload_offset = std::mem::size_of::<AinHeader>();
                            let raw_grads = &read_buf[payload_offset..];
                            let count = raw_grads.len() / 2;
                            
                            // In-place i16 -> f32
                            let ptr = raw_grads.as_ptr() as *const i16;
                            let out_ptr = grads_f32.as_mut_ptr();
                            unsafe {
                                for i in 0..count {
                                    *out_ptr.add(i) = *ptr.add(i) as f32;
                                }
                            }

                            // 4. SECURITY CHECK (The Sentry)
                            // If it fails, we drop it. No GPU work done.
                            if let Err(e) = sentry.validate(header.layer_id, &grads_f32[..count]) {
                                warn!("DROP: {}", e);
                                continue; 
                            }

                            // 5. GPU COMPUTE
                            let dummy_weights = vec![0u8; count * 4];
                            if let Err(e) = gpu.launch_sgd(
                                &dummy_weights, 
                                &grads_f32[..count], 
                                0.01, 
                                count
                            ).await {
                                error!("GPU Launch Failed: {:?}", e);
                            }
                        }
                        // 6. SHUTDOWN
                        _ = signal::ctrl_c() => {
                            info!("Worker {} shutting down", cpu_id);
                            break;
                        }
                    }
                }
            });
        }

        // Wait for Ctrl+C
        tokio::signal::ctrl_c().await?;
        info!("Shutting down...");
        Ok(())
    }
}

// ENTRY POINT 
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let runtime = ReflexRuntime::init().await?;

    // Metrics printer
    let metrics = runtime.metrics.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            interval.tick().await;
            println!("{}", metrics.summary());
        }
    });

    runtime.run().await?;
    Ok(())
}