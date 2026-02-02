
use crate::network::packet::AinHeader;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time::{interval, timeout};
use tracing::{info, warn, error, debug};

const SWARM_PORT: u16 = 13370;
const HEARTBEAT_INTERVAL_MS: u64 = 500;
const STRAGGLER_THRESHOLD_MS: u64 = 500; // 0.5ms is slow for AI

/// Events sent from Swarm to Main Runtime
#[derive(Debug, Clone)]
pub enum SwarmEvent {
    Discovered { id: u32, addr: SocketAddr },
    DataReceived { peer_id: u32, latency: Duration },
    PeerTimeout { id: u32 },
}

/// Represents a live node in the cluster
pub struct Peer {
    id: u32,
    addr: SocketAddr,
    last_seen: Instant,
    avg_latency_ms: u64,
    alive: bool,
}

pub struct Swarm {
    socket: Arc<UdpSocket>,
    peers: HashMap<u32, Peer>,
    my_id: u32,
}

impl Swarm {
    /// Starts the P2P listeners and heartbeat loop.
    pub async fn bootstrap() -> Result<(Self, mpsc::Receiver<SwarmEvent>), anyhow::Error> {
        let socket = Arc::new(UdpSocket::bind("0.0.0.0:13370".parse()?).await?);
        let my_id = rand::random::<u32>();
        
        info!(">> Swarm Bootstrapped. My ID: {}", my_id);

        let (tx, rx) = mpsc::channel(1000);
        let swarm = Self {
            socket,
            peers: HashMap::new(),
            my_id,
        };

        // Task 1: The Listener
        let list_socket = socket.clone();
        let list_tx = tx.clone();
        tokio::spawn(async move {
            let mut buf = [0u8; 1024];
            loop {
                match list_socket.recv_from(&mut buf).await {
                    Ok((len, src)) => {
                        if len < std::mem::size_of::<AinHeader>() { continue; }
                        
                        // Parse Packet (Mock logic for demo)
                        let header = unsafe { &*(buf.as_ptr() as *const AinHeader) };
                        
                        let latency = Duration::from_micros(rand::random::<u64>() % 100);
                        let _ = list_tx.send(SwarmEvent::DataReceived {
                            peer_id: header.node_id,
                            latency,
                        }).await;
                    }
                    Err(e) => error!("Swarm listener error: {:?}", e),
                }
            }
        });

        // Task 2: Heartbeat & Maintenance
        let mut tick = interval(Duration::from_millis(100));
        let peers = swarm.peers.clone(); // We need to clone state or wrap in Mutex.
        // For this demo, we assume single-threaded access for simplicity or use Mutex.
        
        Ok((swarm, rx))
    }

    /// The background loop managing topology.
    /// In a real system, this would run in a tokio::spawn task returned by bootstrap.
    /// Here we assume the user calls this via `swarm.run(rx)`.
    pub async fn run(mut self, mut rx: mpsc::Receiver<SwarmEvent>) -> Result<(), anyhow::Error> {
        let mut check_ticker = interval(Duration::from_millis(HEARTBEAT_INTERVAL_MS));

        loop {
            tokio::select! {
                // 1. Handle Events from Network
                Some(event) = rx.recv() => {
                    match event {
                        SwarmEvent::DataReceived { peer_id, latency } => {
                            self.on_packet_received(peer_id, latency);
                        }
                        _ => {}
                    }
                }
                // 2. Routine Health Check
                _ = check_ticker.tick() => {
                    self.check_timeouts();
                }
                else => {
                    // Shutdown
                    break;
                }
            }
        }
        Ok(())
    }

    fn on_packet_received(&mut self, peer_id: u32, latency: Duration) {
        let now = Instant::now();
        let entry = self.peers.entry(peer_id).or_insert_with(|| Peer {
            id: peer_id,
            addr: "0.0.0.0:0".parse().unwrap(),
            last_seen: now,
            avg_latency_ms: latency.as_millis(),
            alive: true,
        });

        entry.last_seen = now;
        
        // Exponential Moving Average for latency
        let current = entry.avg_latency_ms;
        let new = latency.as_millis();
        entry.avg_latency_ms = (current * 3 + new) / 4; // EMA(0.25)
        
        // Straggler Check
        if entry.avg_latency_ms > STRAGGLER_THRESHOLD_MS {
            warn!("STRAGGLER DETECTED: Node {} ({}ms) is lagging!", peer_id, entry.avg_latency_ms);
            entry.alive = false;
        } else {
            entry.alive = true;
        }
    }

    fn check_timeouts(&mut self) {
        let now = Instant::now();
        let threshold = Duration::from_millis(HEARTBEAT_INTERVAL_MS * 3); // 3 missed heartbeats

        self.peers.retain(|_, peer| {
            if now.duration_since(peer.last_seen) > threshold {
                warn!("Dropping peer: {}", peer.id);
                false // Retain = false, remove it
            } else {
                true
            }
        });
    }
}

// Mock helper for random ID generation
mod rand {
    use std::time::{SystemTime, UNIX_EPOCH};
    pub fn random<T>() -> T where T: From<u64> {
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        T::from(time)
    }
}