

use crate::network::packet::AinHeader;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, timeout, Duration as TokioDuration};
use tracing::{info, warn, error, debug};

const SWARM_PORT: u16 = 13370;
const HEARTBEAT_INTERVAL_MS: u64 = 500;
const BROADCAST_ADDR: &str = "239.255.255.250:13370"; // Multicast LAN
const STRAGGLER_THRESHOLD_MS: u64 = 500; 

/// Events sent to Main Runtime
#[derive(Debug, Clone)]
pub enum SwarmEvent {
    Discovered { id: u32, addr: SocketAddr },
    DataReceived { peer_id: u32, latency: Duration },
    PeerTimeout { id: u32 },
}

/// Represents a live node
#[derive(Debug, Clone)]
pub struct Peer {
    pub id: u32,
    pub addr: SocketAddr,
    pub last_seen: Instant,
    pub avg_latency_ms: u64, // EMA
    pub active: bool,
}

pub struct Swarm {
    socket: Arc<UdpSocket>,
    // V2 CHANGE: Thread-Safe Peer Map
    peers: Arc<RwLock<HashMap<u32, Peer>>>,
    my_id: u32,
}

impl Swarm {
    /// Bootstraps swarm and starts listeners.
    pub async fn bootstrap() -> Result<(Self, mpsc::Receiver<SwarmEvent>), anyhow::Error> {
        let socket = Arc::new(UdpSocket::bind("0.0.0.0:13370".parse()?).await?);
        let my_id = rand::random::<u32>(); // Helper needed
        
        info!(">> Swarm Bootstrapped. My ID: {}", my_id);
        info!(">> Broadcasting on {}", BROADCAST_ADDR);

        let (tx, rx) = mpsc::channel(1000);
        
        let swarm = Self {
            socket,
            peers: Arc::new(RwLock::new(HashMap::new())),
            my_id,
        };

        // Task 1: Listener (Receive Data & Heartbeats)
        let list_socket = socket.clone();
        let list_tx = tx.clone();
        let peers_clone = swarm.peers.clone();
        tokio::spawn(async move {
            let mut buf = [0u8; 1024];
            loop {
                match list_socket.recv_from(&mut buf).await {
                    Ok((len, src)) => {
                        if len < std::mem::size_of::<AinHeader>() { continue; }
                        
                        let header = unsafe { &*(buf.as_ptr() as *const AinHeader) };
                        
                        // Check if this is a heartbeat from a Peer
                        if header.layer_id == 0xFFFF { // ID for Heartbeats
                            let _ = list_tx.send(SwarmEvent::Discovered {
                                id: header.node_id,
                                addr: src
                            }).await;
                        } else {
                            // It's Data
                            let _ = list_tx.send(SwarmEvent::DataReceived {
                                peer_id: header.node_id,
                                latency: Duration::from_millis(10), // Mock
                            }).await;
                        }
                    }
                    Err(e) => error!("Swarm listener error: {:?}", e),
                }
            }
        });

        // Task 2: Heartbeat Broadcaster (I exist!)
        let bcast_socket = socket.clone();
        let peers_map = swarm.peers.clone();
        tokio::spawn(async move {
            let mut ticker = interval(TokioDuration::from_millis(HEARTBEAT_INTERVAL_MS));
            loop {
                ticker.tick().await;
                
                // Construct Heartbeat Packet
                let mut buf = vec![0u8; 24];
                let header = AinHeader {
                    magic: 0x41494E50,
                    node_id: my_id,
                    layer_id: 0xFFFF, // Heartbeat ID
                    chunk_idx: 0,
                    total_chunks: 0,
                    timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64,
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(&header as *const u8 as *const _, buf.as_mut_ptr(), 24);
                }

                // Send to Multicast Address
                let _ = bcast_socket.send_to(&buf, BROADCAST_ADDR.parse().unwrap()).await;
            }
        });

        // Task 3: Cleaner (Drop dead peers)
        let cleaner_peers = swarm.peers.clone();
        tokio::spawn(async move {
            let mut ticker = interval(TokioDuration::from_secs(1));
            loop {
                ticker.tick().await;
                let mut peers = cleaner_peers.write().await;
                let now = Instant::now();
                let count = peers.len();
                peers.retain(|_, p| now.duration_since(p.last_seen) < Duration::from_secs(10));
                let new_count = peers.len();
                drop(peers);
                if new_count < count {
                    info!("[CLEANER] Removed {} dead peers", count - new_count);
                }
            }
        });

        Ok((swarm, rx))
    }

    pub async fn run(mut self, mut event_rx: mpsc::Receiver<SwarmEvent>) -> Result<(), anyhow::Error> {
        let mut ticker = interval(TokioDuration::from_millis(100));

        loop {
            tokio::select! {
                // 1. Handle Network Events
                Some(event) = event_rx.recv() => {
                    self.handle_event(event).await;
                }
                // 2. Routine Tasks
                _ = ticker.tick() => {
                    // Placeholder for other maintenance
                }
            }
        }
    }

    async fn handle_event(&mut self, event: SwarmEvent) {
        match event {
            SwarmEvent::Discovered { id, addr } => {
                let mut peers = self.peers.write().await;
                match peers.entry(id) {
                    std::collections::hash_map::Entry::Occupied(mut p) => {
                        p.last_seen = Instant::now();
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        info!("[SWARM] New Peer: {}", id);
                        e.insert(Peer {
                            id,
                            addr,
                            last_seen: Instant::now(),
                            avg_latency_ms: 0,
                            active: true,
                        });
                    }
                }
            }
            SwarmEvent::DataReceived { peer_id, latency } => {
                let mut peers = self.peers.write().await;
                if let Some(p) = peers.get_mut(&peer_id) {
                    p.last_seen = Instant::now();
                    p.active = true;
                    
                    // Update Latency EMA
                    let current = p.avg_latency_ms as f64;
                    let incoming = latency.as_millis() as f64;
                    p.avg_latency_ms = ((current * 3.0) + incoming) / 4.0;

                    if p.avg_latency_ms > STRAGGLER_THRESHOLD_MS {
                        warn!("[SWARM] STRAGGLER DETECTED: Node {} ({}ms) is lagging!", peer_id, p.avg_latency_ms);
                        p.active = false; // Mark as slow
                    }
                }
            }
            SwarmEvent::PeerTimeout { id } => {
                let mut peers = self.peers.write().await;
                peers.remove(&id);
            }
        }
    }
}

mod rand {
    // Mock random for demo
    use std::time::SystemTime;
    pub fn random<T>() -> T where T: From<u64> {
        let time = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        T::from(time)
    }
}