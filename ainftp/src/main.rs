mod compute;
mod network;
mod consensus;
mod telemetry;

use ainftp_common::{GradientReadyEvent, LayerAccumulator}; // Bring in our strict memory contracts
use aya::{include_bytes_aligned, Bpf, programs::{Xdp, XdpFlags}};
use aya::maps::{AsyncRingBuf, HashMap};
use anyhow::{Context, Result};
use tokio::task::JoinSet;
use tracing::{info, error, warn};
use std::sync::Arc;
use tokio::sync::Mutex;

use compute::{rdma::DmaBufferPool, cuda::GpuEngine};
use network::swarm::Swarm;
use consensus::sentry::AnomalyDetector;
use telemetry::ClusterMetrics;

const GPU_ID: u32 = 0;
const DMA_POOL_SIZE: usize = 64;
const DMA_REGION_SIZE: usize = 1 << 20;
const SENTRY_THRESHOLD: f64 = 3.5;
const MAX_PAYLOAD_SIZE: usize = 64;

pub struct ReflexRuntime {
    bpf: Bpf,
    _pool: DmaBufferPool,
    gpu: Arc<GpuEngine>,
    sentry: Arc<AnomalyDetector>,
    metrics: Arc<ClusterMetrics>,
}

impl ReflexRuntime {
    pub async fn init() -> Result<Self> {
        info!("--- AINFTP v2.0 : ENTERPRISE CONTROL PLANE ---");

        let mut bpf = Bpf::load(include_bytes_aligned!("../../target/bpfel-unknown-none/release/ainftp"))?;
        let prog: &mut Xdp = bpf.program_mut("ainftp").unwrap().try_into()?;
        prog.load()?;
        prog.attach("eth0", XdpFlags::default())?;
        info!(">> Kernel Reflex: ATTACHED & AGGREGATING");

        let (swarm, swarm_rx) = Swarm::bootstrap().await.context("Swarm failed")?;
        tokio::spawn(swarm.run(swarm_rx));
        info!(">> P2P Swarm: ACTIVE");

        let pool = DmaBufferPool::new(DMA_POOL_SIZE, DMA_REGION_SIZE)?;
        let gpu = Arc::new(GpuEngine::new(GPU_ID).await?);
        let sentry = Arc::new(AnomalyDetector::new(SENTRY_THRESHOLD));
        let metrics = Arc::new(ClusterMetrics::new());

        Ok(Self { bpf, _pool: pool, gpu, sentry, metrics })
    }

    pub async fn run(mut self) -> Result<()> {
        // Switch to the high-performance BPF Ring Buffer
        let mut ring_buf = AsyncRingBuf::try_from(self.bpf.map_mut("EVENTS")?)?;
        let mut handlers = JoinSet::new();

        // Pre-allocate GPU memory buffers outside the hot path
        let dummy_weights = vec![0u8; MAX_PAYLOAD_SIZE * 4];
        let mut grads_f32 = vec![0f32; MAX_PAYLOAD_SIZE];

        // Open the shared memory map where the kernel drops the aggregated gradients
        let agg_map: HashMap<_, u64, LayerAccumulator> = HashMap::try_from(self.bpf.map("AGGREGATION_CACHE")?)?;

        handlers.spawn(async move {
            loop {
                // 1. Wait for Kernel Event (Zero CPU spin)
                if let Some(item) = ring_buf.next().await {
                    let event = unsafe { std::ptr::read_unaligned(item.as_ptr() as *const GradientReadyEvent) };
                    
                    let key = ((event.layer_id as u64) << 32) | (event.chunk_idx as u64);

                    // 2. Read aggregated gradients directly from Kernel memory
                    if let Ok(acc) = agg_map.get(&key, 0) {
                        
                        // 3. Fast-path conversion
                        unsafe {
                            let in_ptr = acc.sum_i16.as_ptr();
                            let out_ptr = grads_f32.as_mut_ptr();
                            for i in 0..MAX_PAYLOAD_SIZE {
                                *out_ptr.add(i) = *in_ptr.add(i) as f32;
                            }
                        }

                        // 4. Sentry Check
                        if let Err(e) = self.sentry.validate(event.layer_id, &grads_f32) {
                            warn!("DROP: Byzantine Fault Detected - {}", e);
                            continue;
                        }

                        // 5. Blast to GPU directly
                        if let Err(e) = self.gpu.launch_sgd(&dummy_weights, &grads_f32, 0.01, MAX_PAYLOAD_SIZE).await {
                            error!("CUDA Launch Failed: {:?}", e);
                        }
                    }
                }
            }
        });

        tokio::signal::ctrl_c().await?;
        info!(">> Shutting down cluster node...");
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();
    
    let runtime = ReflexRuntime::init().await?;
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