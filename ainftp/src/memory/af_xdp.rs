use anyhow::{Context, Result};
use std::ffi::CString;
use xsk_rs::{
    config::{SocketConfig, UmemConfig, BindFlags, LibbpfFlags},
    socket::Socket,
    umem::Umem,
    RxQueue, TxQueue, FillQueue, CompQueue,
};

// 64 byte aligned chunk size for the NIC to drop packets into
const UMEM_CHUNK_SIZE: u32 = 2048; 
const UMEM_FRAME_SIZE: u32 = 2048;
const RING_SIZE: u32 = 2048;

pub struct AfXdpBridge {
    _umem: Umem,
    _socket: Socket,
    pub rx: RxQueue,
    pub tx: TxQueue,
    pub fill: FillQueue,
    pub comp: CompQueue,
}

impl AfXdpBridge {
    /// Binds an AF_XDP socket directly to the NIC queue using our pre-allocated DMA pool.
    /// 
    /// SAFETY: `dma_ptr` must be page-aligned, locked in RAM, and valid for the lifetime of this struct.
    pub unsafe fn bind(iface: &str, queue_id: u32, dma_ptr: *mut u8, dma_len: usize) -> Result<Self> {
        let iface_c = CString::new(iface)?;

        // 1. Configure the UMEM (User Memory) region
        let umem_config = UmemConfig::new(
            UMEM_FRAME_SIZE,
            0,
            RING_SIZE,
            RING_SIZE,
            0,
        ).context("Failed to build UMEM config")?;

        // 2. Map our custom GPUDirect DMA region to the NIC
        let (umem, fill, comp) = Umem::new(
            umem_config,
            std::slice::from_raw_parts_mut(dma_ptr, dma_len),
        ).context("Failed to register UMEM with NIC")?;

        // 3. Force Zero Copy at the driver level
        let mut bind_flags = BindFlags::empty();
        bind_flags.insert(BindFlags::XDP_ZEROCOPY); // Hard fail if NIC doesn't support zero-copy
        bind_flags.insert(BindFlags::XDP_USE_NEED_WAKEUP);

        let socket_config = SocketConfig::new(
            RING_SIZE,
            RING_SIZE,
            bind_flags,
            LibbpfFlags::empty(),
            0,
        ).context("Failed to build Socket config")?;

        // 4. Bind the socket to the hardware queue
        let (socket, rx, tx) = Socket::new(
            socket_config,
            &umem,
            &iface_c,
            queue_id,
        ).context("AF_XDP Socket bind failed. Is the interface down?")?;

        // 5. Pre fill the NIC's receive ring with addresses from our DMA pool
        let mut fill_q = fill;
        let frames_to_fill = dma_len as u64 / UMEM_CHUNK_SIZE as u64;
        
        unsafe {
            fill_q.produce(frames_to_fill as usize, |frame| {
                // Pass raw physical offsets to the NIC
                frame.set_addr(frame.index() as u64 * UMEM_CHUNK_SIZE as u64);
            });
        }

        Ok(Self {
            _umem: umem,
            _socket: socket,
            rx,
            tx,
            fill: fill_q,
            comp,
        })
    }

    /// Polls the Rx ring for new hardware-aggregated events
    #[inline(always)]
    pub fn poll_rx(&mut self, batch_size: usize) -> Vec<u64> {
        let mut offsets = Vec::with_capacity(batch_size);
        
        unsafe {
            self.rx.consume(batch_size, |desc| {
                offsets.push(desc.addr());
            });
        }
        
        offsets
    }

    /// Returns consumed frames back to the NIC hardware
    #[inline(always)]
    pub fn recycle_frames(&mut self, offsets: &[u64]) {
        unsafe {
            self.fill.produce(offsets.len(), |frame| {
                frame.set_addr(offsets[frame.index()]);
            });
        }
    }
}