//! P2P Swarm Manager
//! 
//! Maintains the list of active nodes.
//! - Heartbeats: Keeps the topology fresh.
//! - Straggler Detection: If a node > 200ms latency, we ignore it.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::net::UdpSocket;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug};
use crate::network::packet::{AinHeader, NetworkPacket, PacketType};

/// Configuration for the Swarm
const SWARM_PORT: u16 = 13370;
const HEARTBEAT_INTERVAL_MS: u64 = 1000;
const STRAGGLER_THRESHOLD_MS: u64 = 500; // 0.5ms is slow for AI

#[derive(Debug, Clone)]
pub struct Peer {
    pub id: u32,
    pub addr: SocketAddr,
    pub last_seen: Instant,
    pub latency_ms: u64,
    pub active: bool,
}

/// Messages sent internally from the Swarm listener to the Logic loop
#[derive(Debug)]
pub enum SwarmEvent {
    Discovered { id: u32, addr: SocketAddr },
    DataReceived { peer_id: u32, layer_id: u32, latency: Duration },
    PeerTimeout { id: u32 },
}

pub struct Swarm {
    peers: HashMap<u32, Peer>,
    socket: UdpSocket,
    my_id: u32,
}

impl Swarm {
    /// Starts the P2P Swarm.
    /// This spawns a background task to listen for heartbeats.
    pub async fn bootstrap() -> Result<(Self, mpsc::Receiver<SwarmEvent>)> {
        // 1. Bind Transport
        let transport = crate::network::transport::AinTransport::bind(SWARM_PORT)?;
        let socket = transport.into_tokio()?;

        let my_id = rand::random::<u32>(); // Random Node ID
        info!("Swarm Initialized. My Node ID: {}", my_id);

        let swarm = Self {
            peers: HashMap::new(),
            socket,
            my_id,
        };

        // 2. Start Background Listener
        let (tx, rx) = mpsc::channel(1000);
        let socket_clone = swarm.socket.try_clone()?;
        
        tokio::spawn(async move {
            let mut buf = [0u8; 1024];
            loop {
                match socket_clone.recv_from(&mut buf).await {
                    Ok((len, src)) => {
                        // Simplified parsing. In real code, we check PacketType.
                        // Here we assume every packet is a heartbeat or data.
                        
                        // Parse Header (mock logic)
                        if len < 24 { continue; } // Min header size
                        
                        // In a real impl, we'd deserialize `AinHeader` here.
                        // For now, we simulate discovery.
                        let peer_id = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
                        let layer_id = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);

                        let _ = tx.send(SwarmEvent::DataReceived {
                            peer_id,
                            layer_id,
                            latency: Duration::from_millis(10), // Mock
                        }).await;
                    }
                    Err(e) => {
                        warn!("Swarm listener error: {:?}", e);
                    }
                }
            }
        });

        Ok((swarm, rx))
    }

    /// The main logic loop.
    /// This is where we manage the state of the cluster.
    pub async fn run(mut self, mut event_rx: mpsc::Receiver<SwarmEvent>) {
        let mut ticker = tokio::time::interval(Duration::from_millis(HEARTBEAT_INTERVAL_MS));

        loop {
            tokio::select! {
                // 1. Handle Incoming Events
                Some(event) = event_rx.recv() => {
                    match event {
                        SwarmEvent::DataReceived { peer_id, layer_id, latency } => {
                            // Update Latency
                            if let Some(peer) = self.peers.get_mut(&peer_id) {
                                peer.last_seen = Instant::now();
                                peer.latency_ms = latency.as_millis() as u64;
                                
                                // STRAGGLER LOGIC
                                if peer.latency_ms > STRAGGLER_THRESHOLD_MS {
                                    warn!("STRAGGLER DETECTED: Node {} is lagging ({}ms). Degrading.", peer_id, peer.latency_ms);
                                    peer.active = false;
                                } else {
                                    peer.active = true;
                                }
                            } else {
                                // New Peer
                                info!("New peer joined: {}", peer_id);
                                // We'd send a Handshake back here
                            }
                        }
                        _ => {}
                    }
                }
                
                // 2. Periodic Maintenance (Garbage Collection)
                _ = ticker.tick() => {
                    let now = Instant::now();
                    // Remove dead peers
                    self.peers.retain(|_, p| now.duration_since(p.last_seen) < Duration::from_secs(10));
                    debug!("Swarm Size: {} active nodes", self.peers.len());
                }
            }
        }
    }

    pub fn get_peers(&self) -> Vec<u32> {
        self.peers.keys().cloned().collect()
    }
}