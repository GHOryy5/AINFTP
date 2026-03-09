mod compute;
mod network;
mod consensus;
mod telemetry;

use aya::{include_bytes_aligned, Bpf, programs::{Xdp, XdpFlags}, maps::RingBuf};
use aya::util::online_cpus;
use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio::signal;
use tracing::{info, error, warn, debug};

use compute::{rdma::DmaBufferPool, cuda::GpuEngine};
use network::{swarm::Swarm, packet::AinHeader}; 
use consensus::sentry::AnomalyDetector;
use telemetry::ClusterMetrics;

const GPU_ID: u32 = 0;
const DMA_POOL_SIZE: usize = 128;
const DMA_REGION_SIZE: usize = 1024 * 1024;
const SENTRY_THRESHOLD: f64 = 3.5;  

pub struct ReflexRuntime {
    _bpf: Bpf,
    _pool: DmaBufferPool,
    gpu: GpuEngine,
    sentry: AnomalyDetector,
    _swarm_rx: mpsc::Receiver<network::swarm::SwarmEvent>, 
    metrics: ClusterMetrics,
}

impl ReflexRuntime {
    pub async fn init() -> Result<Self> {
        info!("Initializing ainftp v2.1 [Production Engine]");

        let mut bpf = Bpf::load(include_bytes_aligned!(
            "../../target/bpfel-unknown-none/release/ainftp"
        ))?;
        let prog: &mut Xdp = bpf.program_mut("ainftp").unwrap().try_into()?;
        prog.load()?;
        prog.attach("eth0", XdpFlags::default())?; 

        let (swarm, swarm_rx) = Swarm::bootstrap().await?;
        tokio::spawn(swarm.run(swarm_rx.clone())); 

        let pool = DmaBufferPool::new(DMA_POOL_SIZE, DMA_REGION_SIZE)?;
        let gpu = GpuEngine::new(GPU_ID)?;
        let sentry = AnomalyDetector::new(SENTRY_THRESHOLD);
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

    pub async fn run(mut self) -> Result<()> {
        let events = self._bpf.map("EVENTS")?;
        let mut ring = RingBuf::try_from(events)?;
        let mut handlers = JoinSet::new();

        for cpu_id in online_cpus()? {
            let gpu = self.gpu.clone();
            let mut sentry = self.sentry.clone();
            let mut ring_reader = ring.clone();
            let metrics = self.metrics.clone();
            
            handlers.spawn(async move {
                let mut grads_f32 = vec![0f32; 32768];
                info!("Worker CPU {} started [Zero-Copy Active]", cpu_id);

                loop {
                    // Fast polling with async yield
                    while let Some(item) = ring_reader.next() {
                        let data = &*item;
                        if data.len() < std::mem::size_of::<AinHeader>() { continue; }

                        let header = unsafe { &*(data.as_ptr() as *const AinHeader) };
                        let payload_offset = std::mem::size_of::<AinHeader>();
                        let raw_grads = &data[payload_offset..];
                        let count = raw_grads.len() / 2;

                        // Vectorized conversion
                        let ptr = raw_grads.as_ptr() as *const i16;
                        for i in 0..count {
                            unsafe { *grads_f32.get_unchecked_mut(i) = *ptr.add(i) as f32; }
                        }

                        // Sentry Filter
                        if let Err(e) = sentry.validate(header.layer_id, &grads_f32[..count]) {
                            warn!("Sentry rejection: {:?}", e);
                            continue;
                        }

                        // Direct GPU injection
                        if let Err(e) = gpu.launch_sgd(
                            0,
                            grads_f32.as_ptr() as u64,
                            0.01,
                            count
                        ).await {
                            error!("CUDA Failure: {:?}", e);
                        }

                        metrics.packets_rx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    tokio::task::yield_now().await;
                }
            });
        }

        let metrics_clone = self.metrics.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
            loop {
                interval.tick().await;
                println!("{}", metrics_clone.summary());
            }
        });

        signal::ctrl_c().await?;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let runtime = ReflexRuntime::init().await?;
    runtime.run().await
}
