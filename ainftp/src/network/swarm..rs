use ainftp_common::AinftpHeader; // Use our strict memory contract
use anyhow::{Result, Context};
use std::collections::HashMap;
use std::net::{SocketAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, Duration as TokioDuration};
use tracing::{info, warn, error, debug};
use socket2::{Socket, Domain, Type, Protocol};

const SWARM_PORT: u16 = 13370;
const HEARTBEAT_INTERVAL_MS: u64 = 500;
const PEER_TIMEOUT_SECS: u64 = 5;
// Multicast group for peer discovery on the local switch
const MULTICAST_IP: &str = "239.255.255.250"; 

#[derive(Debug, Clone)]
pub enum SwarmEvent {
    PeerDiscovered { id: u32, addr: SocketAddr },
    PeerLost { id: u32 },
}

#[derive(Debug, Clone)]
pub struct Peer {
    pub id: u32,
    pub addr: SocketAddr,
    pub last_seen: Instant,
    pub active: bool,
}

pub struct Swarm {
    socket: Arc<UdpSocket>,
    peers: Arc<RwLock<HashMap<u32, Peer>>>,
    my_id: u32,
}

impl Swarm {
    pub async fn bootstrap() -> Result<(Self, mpsc::Receiver<SwarmEvent>)> {
        let my_id = std::process::id(); // Use PID as a fast, unique node ID for now
        
        info!(">> Bootstrapping Control Plane Swarm. Node ID: {}", my_id);

        // 1. Setup Multicast Socket via socket2 to ensure proper IGMP joins
        let multi_addr: Ipv4Addr = MULTICAST_IP.parse()?;
        let bind_addr: SocketAddr = format!("0.0.0.0:{}", SWARM_PORT).parse()?;

        let sock = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
        sock.set_reuse_address(true)?;
        sock.set_reuse_port(true)?; // Required for multiple nodes on same machine
        sock.bind(&bind_addr.into())?;
        sock.join_multicast_v4(&multi_addr, &Ipv4Addr::UNSPECIFIED)?;

        let socket = Arc::new(UdpSocket::from_std(sock.into())?);
        let (tx, rx) = mpsc::channel(100);
        let peers = Arc::new(RwLock::new(HashMap::new()));

        let swarm = Self {
            socket: socket.clone(),
            peers: peers.clone(),
            my_id,
        };

        // Task 1: Control Plane Listener (HEARTBEATS ONLY)
        // Data packets (layer_id != 0xFFFF) are handled by AF_XDP/eBPF. i ignore them here.
        let list_socket = socket.clone();
        let list_tx = tx.clone();
        let list_peers = peers.clone();
        
        tokio::spawn(async move {
            let mut buf = [0u8; 64]; // Small buffer, we only expect headers here
            loop {
                match list_socket.recv_from(&mut buf).await {
                    Ok((len, src)) => {
                        if len < std::mem::size_of::<AinftpHeader>() { continue; }
                        
                        let header = unsafe { &*(buf.as_ptr() as *const AinftpHeader) };
                        
                        // magic check
                        if header.magic != 0x4149 { continue; }

                        // Is it a heartbeat? (Using flags or a specific sequence for metadata)
                        if header.flags == 0xFF {
                            let mut p_map = list_peers.write().await;
                            
                            if !p_map.contains_key(&header.node_id) && header.node_id != my_id {
                                info!(">> Topology Update: Discovered Node {} at {}", header.node_id, src);
                                let _ = list_tx.send(SwarmEvent::PeerDiscovered {
                                    id: header.node_id,
                                    addr: src,
                                }).await;
                            }

                            p_map.insert(header.node_id, Peer {
                                id: header.node_id,
                                addr: src,
                                last_seen: Instant::now(),
                                active: true,
                            });
                        }
                    }
                    Err(e) => error!("Control Plane Listener Error: {:?}", e),
                }
            }
        });

        // Task 2: Heartbeat Emitter
        let bcast_socket = socket.clone();
        tokio::spawn(async move {
            let mut ticker = interval(TokioDuration::from_millis(HEARTBEAT_INTERVAL_MS));
            let bcast_addr: SocketAddr = format!("{}:{}", MULTICAST_IP, SWARM_PORT).parse().unwrap();
            
            loop {
                ticker.tick().await;
                
                let header = AinftpHeader {
                    magic: 0x4149,
                    version: 1,
                    flags: 0xFF, // 0xFF denotes Control Plane metadata
                    node_id: my_id,
                    sequence: 0,
                };

                let buf: [u8; std::mem::size_of::<AinftpHeader>()] = unsafe { std::mem::transmute(header) };
                let _ = bcast_socket.send_to(&buf, bcast_addr).await;
            }
        });

        // Task 3: The Reaper (Prunes dead connections)
        let reaper_peers = peers.clone();
        let reaper_tx = tx.clone();
        tokio::spawn(async move {
            let mut ticker = interval(TokioDuration::from_secs(1));
            loop {
                ticker.tick().await;
                let mut p_map = reaper_peers.write().await;
                let now = Instant::now();
                
                let mut dead_peers = Vec::new();
                for (id, peer) in p_map.iter() {
                    if now.duration_since(peer.last_seen) > Duration::from_secs(PEER_TIMEOUT_SECS) {
                        dead_peers.push(*id);
                    }
                }

                for id in dead_peers {
                    warn!(">> Topology Update: Node {} timed out. Evicting.", id);
                    p_map.remove(&id);
                    let _ = reaper_tx.send(SwarmEvent::PeerLost { id }).await;
                    
                    // TODO: Trigger an eBPF map update here to remove them from ALLOWED_NODES
                }
            }
        });

        Ok((swarm, rx))
    }

    pub async fn run(self, mut event_rx: mpsc::Receiver<SwarmEvent>) {
        loop {
            if let Some(event) = event_rx.recv().await {
                match event {
                    SwarmEvent::PeerDiscovered { id, addr } => {
                        debug!("Control Plane processing new peer: {} ({})", id, addr);
                        // Future: Sync consensus state with new peer
                    }
                    SwarmEvent::PeerLost { id } => {
                        debug!("Control Plane evicted peer: {}", id);
                        // Future: Trigger re-sharding of layer matrix
                    }
                }
            }
        }
    }
}