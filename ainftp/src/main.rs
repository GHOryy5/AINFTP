//! ainftp: The Distributed AI Reflex
//! 
//! The Conductor.
//! 
//! Architecture:
//! 1. XDP/eBPF (Kernel) -> Aggregates packets -> Sends to RingBuffer.
//! 2. Runtime (User) -> Reads RingBuffer -> Validates (Sentry) -> Allocates (RDMA) -> Launches (CUDA).
//! 3. Swarm (Network) -> Manages Peers & Topology.

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
use tracing::{info, error, debug, warn};

// Import our "God Tier" modules
use compute::{rdma::DmaBufferPool, cuda::GpuEngine};
use network::{swarm::Swarm, packet::AinHeader}; 
use consensus::sentry::AnomalyDetector;
use telemetry::ClusterMetrics;

// --- CONFIGURATION ---
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
    
    // 3. Intelligence
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
        // This binary comes from xtask
        let bpf = Bpf::load(include_bytes_aligned!(
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

        // Spawn a worker thread for each CPU core
        for cpu_id in online_cpus()? {
            let mut buf_reader = perf_array.open(cpu_id, None)?;
            let gpu = self.gpu.clone();
            
            handlers.spawn(async move {
                let mut buffers = vec![vec![0u8; 1024]];
                
                loop {
                    // 1. READ FROM KERNEL
                    match buf_reader.read_events(&mut buffers) {
                        Ok(events) => {
                            for buf in &buffers {
                                if buf.len() < std::mem::size_of::<AinHeader>() { continue; } 

                                // 2. PARSE HEADER
                                let header = unsafe { &*(buf.as_ptr() as *const AinHeader) };
                                debug!("RX Layer {} Chunk {}", header.layer_id, header.chunk_idx);

                                // 3. PARSE PAYLOAD (i16 -> f32)
                                let payload_offset = std::mem::size_of::<AinHeader>();
                                let raw_grads = &buf[payload_offset..];
                                
                                // Convert kernel i16 data to f32 for processing
                                let grads_f32: Vec<f32> = raw_grads
                                    .chunks_exact(2)
                                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
                                    .collect();

                                // 4. SECURITY CHECK (The Sentry)
                                // Pass gradients to the Sentry to check for Poison
                                if let Err(e) = gpu.validate_sentry(header.layer_id, &grads_f32) {
                                    error!("!!! SECURITY ALERT: {} !!!", e);
                                    continue; // Drop the packet
                                }

                                // 5. GPU COMPUTE
                                // We pass the gradients to the GPU.
                                // Note: weights_ptr is mocked here as we don't have a Model Loader module.
                                let dummy_weights = vec![0u8; grads_f32.len() * 4];
                                
                                if let Err(e) = gpu.launch_sgd(
                                    &dummy_weights, 
                                    &grads_f32, 
                                    0.01, // Learning Rate
                                    grads_f32.len()
                                ).await {
                                    error!("GPU Launch Failed: {:?}", e);
                                }
                            }
                        }
                        Err(e) => {
                            if e.kind() != std::io::ErrorKind::WouldBlock {
                                error!("Perf Read Error: {:?}", e);
                            }
                        }
                    }
                }
            });
        }

        // Wait for Ctrl+C or crash
        tokio::signal::ctrl_c().await.ok();
        info!("Shutting down...");
        Ok(())
    }
}

// --- ENTRY POINT ---
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