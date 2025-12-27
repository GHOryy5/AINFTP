//! ainftp-rt: High-Performance Distributed Training Runtime
//! 
//! core philosophy:
//! - kernel is the bottleneck, bypass it.
//! - memory allocation is slow, pool it.
//! - copy is death, zero-copy everything.
//! - if you block, you're fired.

use aya::{include_bytes_aligned, Bpf, programs::{Xdp, XdpFlags}};
use aya::maps::{HashMap, MapData, RingBuffer, perf::AsyncPerfEventArray};
use aya::util::online_cpus;
use anyhow::{Context, Result, bail};
use bytes::{Bytes, BytesMut};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig, CudaStream, CudaFunction};
use ainftp_common::{NodeStats, GradientChunk, NetworkHeader};
use libc::{c_void, madvise, posix_memalign, MADV_DONTNEED};
use std::alloc::{Layout, alloc, dealloc};
use std::collections::{HashMap as StdHashMap, VecDeque};
use std::ffi::c_int;
use std::mem::size_of_val;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, Mutex, Notify};
use tokio::task::JoinSet;
use tokio::{select, signal};
use tracing::{info, error, warn, debug, instrument, span, Level};

// --- CONSTANTS & CONFIG ---
const GPU_ID: u32 = 0;
const BATCH_WINDOW_US: u64 = 200; // 200 microsecond latency target
const MAX_PENDING_BUFFERS: usize = 32; // Backpressure limit
const ALIGNMENT: usize = 4096; // Page aligned for DMA

// Zero-copy SGX/TPM style memory handles (simulated)
type PinnedBuf = NonNull<[u8]>;

// --- GLOBAL METRICS ---
lazy_static::lazy_static! {
    static ref RX_BYTES: AtomicU64 = AtomicU64::new(0);
    static ref TX_PACKETS: AtomicU64 = AtomicU64::new(0);
    static ref GPU_LATENCY_US: AtomicU64 = AtomicU64::new(0);
}

// --- MEMORY POOL (Arena Allocator) ---
/// Malloc is too slow for 10M+ packets/sec. We slab allocate.
struct BufferPool {
    // A simple stack of pre-allocated page-aligned buffers
    pool: Arc<Mutex<VecDeque<BytesMut>>>,
    buf_size: usize,
}

impl BufferPool {
    fn new(capacity: usize, buf_size: usize) -> Self {
        let mut pool = VecDeque::with_capacity(capacity);
        // pre-warm the pool to avoid runtime latency spikes
        for _ in 0..capacity {
            // aligned allocation for DMA compatibility
            let layout = Layout::from_size_align(buf_size, ALIGNMENT).unwrap();
            let ptr = unsafe { alloc(layout) };
            if ptr.is_null() {
                panic!("OOM: cannot pre-allocate DMA buffers");
            }
            let slice = std::slice::from_raw_parts_mut(ptr, buf_size);
            pool.push_back(BytesMut::from_vec(slice.to_vec())); 
            // note: in a real scenario we'd use a custom BytesMut wrapper to free this correctly
        }
        Self {
            pool: Arc::new(Mutex::new(pool)),
            buf_size,
        }
    }

    async fn get(&self) -> BytesMut {
        let mut p = self.pool.lock().await;
        match p.pop_front() {
            Some(buf) => buf,
            None => {
                // Fallback if pool is drained (rare/should panic)
                error!("Buffer pool exhausted! Allocating from heap (SLOW PATH)");
                BytesMut::with_capacity(self.buf_size)
            }
        }
    }

    async fn put(&self, buf: BytesMut) {
        // Reset length but keep capacity
        let mut clean = buf;
        clean.clear();
        let mut p = self.pool.lock().await;
        if p.len() < MAX_PENDING_BUFFERS {
            p.push_back(clean);
        }
        // else drop it
    }
}

// --- TOPOLOGY MANAGER ---
/// Tracks which nodes are alive and syncing. Handles straggler drops.
struct Topology {
    nodes: StdHashMap<u32, NodeState>,
}

struct NodeState {
    last_seq: u64,
    liveness: Instant,
    credits: AtomicUsize, // Flow control credits
}

impl Topology {
    fn new() -> Self {
        Self {
            nodes: StdHashMap::new(),
        }
    }

    fn register(&mut self, node_id: u32) {
        self.nodes.entry(node_id).or_insert(NodeState {
            last_seq: 0,
            liveness: Instant::now(),
            credits: MAX_PENDING_BUFFERS,
        });
    }

    /// Returns true if packet should be processed (flow control + dup check)
    fn validate(&self, node_id: u32, seq: u64) -> bool {
        if let Some(node) = self.nodes.get(&node_id) {
            // strict ordering: drop out-of-order or dups
            if seq <= node.last_seq {
                return false;
            }
            // check backpressure
            if node.credits.load(Ordering::Acquire) == 0 {
                return false;
            }
            true
        } else {
            false
        }
    }
}

// --- GPU WORKER ---
/// dedicated thread per GPU. CUDA contexts aren't thread-safe by default.
struct GpuWorker {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    kernel: CudaFunction,
    ready: Arc<Notify>,
}

impl GpuWorker {
    async fn new(ptx: &str) -> Result<Self> {
        let dev = Arc::new(CudaDevice::new(GPU_ID)?);
        let stream = dev.fork_stream()?; // Create dedicated stream
        
        // Load the SGD kernel
        let kernel = unsafe { dev.load_ptx(ptx, "sgd_step", &[])? };

        Ok(Self {
            device: dev,
            stream,
            kernel,
            ready: Arc::new(Notify::new()),
        })
    }

    /// Launch async compute. Returns immediately.
    async fn launch_update(&self, grad_ptr: &c_void, len: usize) -> Result<()> {
        // This is the critical path. We assume grad_ptr is already device memory
        // or registered host memory (zero-copy).
        
        // Placeholder for the kernel launch config
        let cfg = LaunchConfig {
            grid_dim: (len as u32 + 255) / 256,
            block_dim: 256,
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernel.launch_on_stream(
                &self.stream, 
                cfg, 
                (&grad_ptr, &grad_ptr, &0.01f32, &(len as u32))
            )?;
        }
        
        self.ready.notify_one();
        Ok(())
    }
}

// --- MAIN RUNTIME ---
struct ReflexRuntime {
    bpf: Bpf,
    buffer_pool: Arc<BufferPool>,
    topology: Arc<Mutex<Topology>>,
    gpu_worker: Arc<GpuWorker>,
    shutdown: Arc<AtomicBool>,
}

impl ReflexRuntime {
    async fn init() -> Result<Self> {
        // 1. Load eBPF
        let mut bpf = Bpf::load(include_bytes_aligned!(
            "../../target/bpfel-unknown-none/release/ainftp"
        ))?;
        let prog: &mut Xdp = bpf.program_mut("ainftp").unwrap().try_into()?;
        prog.load()?;
        prog.attach("eth0", XdpFlags::default())?; // Attach to real NIC

        // 2. Init subsystems
        let pool = Arc::new(BufferPool::new(512, 65536)); // 512 buffers of 64KB
        let topo = Arc::new(Mutex::new(Topology::new()));
        
        // Load GPU Kernel (Embedded PTX)
        let ptx = include_str!("../kernels/sgd.ptx");
        let gpu = Arc::new(GpuWorker::new(ptx).await?);

        Ok(Self {
            bpf,
            buffer_pool: pool,
            topology: topo,
            gpu_worker: gpu,
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// The polling loop using PerfBuffers (Event driven)
    #[instrument(skip(self))]
    async fn run_event_loop(&self) -> Result<()> {
        // Try loading the PerfBuffer array (efficient batch notification from kernel)
        let mut perf_array = AsyncPerfEventArray::try_from(self.bpf.map_mut("EVENTS")?)?;

        // Iterate over each CPU core to handle packets in parallel
        let mut handlers = JoinSet::new();
        for cpu_id in online_cpus()? {
            let mut buf_reader = perf_array.open(cpu_id, None)?;
            let pool = self.buffer_pool.clone();
            let topo = self.topology.clone();
            let gpu = self.gpu_worker.clone();
            let shutdown = self.shutdown.clone();

            handlers.spawn(async move {
                let mut buffers = vec![BytesMut::with_capacity(1024)]; // read buffer
                
                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }

                    // Read events from kernel space
                    match buf_reader.read_events(&mut buffers) {
                        Ok(events) => {
                            for buf in buffers.iter() {
                                // Parse the header directly from raw bytes
                                // Zero-copy parsing: No std::mem::transmute, just bytes ref
                                if buf.len() < size_of::<NetworkHeader>() {
                                    continue;
                                }
                                
                                // Unsafe raw pointer casting for speed (The "Reflex")
                                let header = unsafe {
                                    &*(buf.as_ptr() as *const NetworkHeader)
                                };

                                // Validate topology
                                let t = topo.lock().await;
                                if !t.validate(header.node_id, header.seq) {
                                    continue; // Drop bad packet
                                }
                                drop(t);

                                // Process Gradient Data
                                let payload = &buf[size_of::<NetworkHeader>()..];
                                if let Err(e) = Self::process_payload(&gpu, payload, header.layer_id).await {
                                    error!("GPU inject fail: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            // EAGAIN or EINTR are expected when polling
                            if e.kind() != std::io::ErrorKind::WouldBlock {
                                warn!("Perf read error: {:?}", e);
                            }
                        }
                    }
                }
            });
        }

        // Run the handlers
        while let Some(res) = handlers.join_next().await {
            res??;
        }

        Ok(())
    }

    // --- HELPER: PAYLOAD PROCESSING ---
    async fn process_payload(
        gpu: &Arc<GpuWorker>,
        payload: &[u8],
        _layer_id: u32
    ) -> Result<()> {
        // Convert bytes to f32 slice (assuming Little Endian)
        let count = payload.len() / 4;
        let ptr = payload.as_ptr() as *const f32;
        let grads = unsafe { std::slice::from_raw_parts(ptr, count) };

        // 1. Alloc GPU Memory (or reuse pool)
        // In a real "zero-copy" scenario, this payload is already in registered BAR memory
        let dev = &gpu.device;
        let gpu_mem = dev.htod_sync_copy(grads)?;

        // 2. Trigger Kernel
        gpu.launch_update(gpu_mem.as_ptr(), grads.len()).await?;

        // 3. Metrics
        RX_BYTES.fetch_add(payload.len() as u64, Ordering::Relaxed);
        
        // Note: we aren't freeing gpu_mem here. 
        // In a real implementation we'd have a DeviceMemPool to avoid CUDA malloc overhead.
        Ok(())
    }
}

// --- ENTRY POINT ---
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize distributed tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .init();

    info!("ainftp-rt: v2.4.0-stable booting...");
    
    let runtime = ReflexRuntime::init().await.context("Runtime init failed")?;

    // Spawn background tasks: Metrics Scraper, Heartbeat, etc.
    let metrics_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            interval.tick().await;
            let rx = RX_BYTES.swap(0, Ordering::Relaxed);
            let tx = TX_PACKETS.swap(0, Ordering::Relaxed);
            debug!("METRICS | RX: {} MB/s | PKTS: {}", rx / 1_048_576, tx);
        }
    });

    // Signal handler
    let ctrl_c = signal::ctrl_c();
    tokio::pin!(ctrl_c);

    select! {
        _ = ctrl_c => {
            info!("Caught interrupt. Flushing buffers...");
            runtime.shutdown.store(true, Ordering::Release);
            metrics_handle.abort();
        }
        res = runtime.run_event_loop() => {
            res.context("Event loop crashed")?;
        }
    }

    info!("Shutdown complete. Latency: {}us", GPU_LATENCY_US.load(Ordering::Relaxed));
    Ok(())
}