//! High-Performance Transport Layer
//! 
//! Uses `socket2` to tune the kernel buffers and enable SO_REUSEPORT.
//! This allows multiple threads to listen on the same port for load balancing.

use socket2::{Socket, Domain, Type, Protocol};
use std::net::{SocketAddr, Ipv4Addr};
use anyhow::{Result, bail};
use tracing::info;

pub struct AinTransport {
    socket: Socket,
}

impl AinTransport {
    /// Binds to the UDP port with massive buffers.
    /// 
    /// # Args
    /// * `port`: The UDP port to listen on (e.g., 0xA1N or 1337).
    pub fn bind(port: u16) -> Result<Self> {
        let addr = SocketAddr::new(Ipv4Addr::new(0, 0, 0, 0).into(), port);

        // 1. Create UDP Socket
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;

        // 2. OPTIMIZATION: SO_REUSEPORT
        // Allows multiple threads/cores to accept packets on the same socket.
        // Critical for NUMA performance.
        socket.set_reuse_port(true)?;

        // 3. OPTIMIZATION: Increase Kernel Buffer Size
        // Default is usually 128KB. We set it to 16MB to handle bursts without packet loss.
        const RCV_BUF_SIZE: usize = 16 * 1024 * 1024; 
        socket.set_recv_buffer_size(RCV_BUF_SIZE)?;
        socket.set_send_buffer_size(RCV_BUF_SIZE)?;

        // 4. Bind
        socket.bind(&addr.into())?;

        info!("Transport bound to {} with {}MB buffers", port, RCV_BUF_SIZE / 1_048_576);

        Ok(Self { socket })
    }

    /// Convert to Tokio UdpSocket for async usage
    pub fn into_tokio(self) -> Result<tokio::net::UdpSocket> {
        // Convert std::net::UdpSocket to tokio
        let std_sock: std::net::UdpSocket = self.socket.into();
        std_sock.set_nonblocking(true)?;
        Ok(tokio::net::UdpSocket::from_std(std_sock)?)
    }
}