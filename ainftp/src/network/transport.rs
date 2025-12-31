//! High-Performance Transport Layer
//! Uses `socket2` to tune the kernel buffers and enable SO_REUSEPORT.
//! This allows multiple threads to listen on the same port for load balancing.

use socket2::{Socket, Domain, Type, Protocol};
use std::net::{SocketAddr, Ipv4Addr};
use anyhow::{Result, bail, Context};
use tracing::{info, warn};
use std::env;

pub struct AinTransport {
    socket: Socket,
}

impl AinTransport {
    /// Transport configuration controls safe defaults and limits for bind behavior.
    pub struct TransportConfig {
        /// Optional explicit bind address. If None, defaults to loopback.
        pub bind_addr: Option<SocketAddr>,
        /// Requested receive buffer size in bytes.
        pub recv_buffer_size: usize,
        /// Requested send buffer size in bytes.
        pub send_buffer_size: usize,
        /// Whether to request SO_REUSEPORT. If unavailable, we fall back.
        pub reuse_port: bool,
    }

    impl Default for TransportConfig {
        fn default() -> Self {
            Self {
                bind_addr: None,
                recv_buffer_size: 4 * 1024 * 1024, // safer default than extremely large
                send_buffer_size: 4 * 1024 * 1024,
                reuse_port: true,
            }
        }
    }

    /// Builder for `TransportConfig` providing a fluent, validated constructor.
    pub struct TransportConfigBuilder {
        bind_addr: Option<SocketAddr>,
        bind_all: bool,
        recv_buffer_size: usize,
        send_buffer_size: usize,
        reuse_port: bool,
    }

    impl Default for TransportConfigBuilder {
        fn default() -> Self {
            Self {
                bind_addr: None,
                bind_all: false,
                recv_buffer_size: TransportConfig::default().recv_buffer_size,
                send_buffer_size: TransportConfig::default().send_buffer_size,
                reuse_port: true,
            }
        }
    }

    impl TransportConfig {
        /// Create a builder for `TransportConfig`.
        pub fn builder() -> TransportConfigBuilder {
            TransportConfigBuilder::default()
        }
    }

    #[derive(Debug)]
    pub enum TransportError {
        PrivilegedPort,
        InvalidBufferSize { requested: usize, max: usize },
    }

    impl std::fmt::Display for TransportError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                TransportError::PrivilegedPort => write!(f, "binding to privileged port requires elevation"),
                TransportError::InvalidBufferSize { requested, max } => write!(f, "requested buffer {} exceeds max {}", requested, max),
            }
        }
    }

    impl std::error::Error for TransportError {}

    impl TransportConfigBuilder {
        pub fn bind_addr(mut self, addr: SocketAddr) -> Self {
            self.bind_addr = Some(addr);
            self
        }

        /// Explicitly request bind on 0.0.0.0
        pub fn bind_all(mut self, yes: bool) -> Self {
            self.bind_all = yes;
            self
        }

        pub fn recv_buffer(mut self, bytes: usize) -> Self {
            self.recv_buffer_size = bytes;
            self
        }

        pub fn send_buffer(mut self, bytes: usize) -> Self {
            self.send_buffer_size = bytes;
            self
        }

        pub fn reuse_port(mut self, yes: bool) -> Self {
            self.reuse_port = yes;
            self
        }

        pub fn build(self) -> Result<TransportConfig, TransportError> {
            const MAX_BUFFER_SIZE: usize = 16 * 1024 * 1024;
            if self.recv_buffer_size > MAX_BUFFER_SIZE {
                return Err(TransportError::InvalidBufferSize { requested: self.recv_buffer_size, max: MAX_BUFFER_SIZE });
            }
            if self.send_buffer_size > MAX_BUFFER_SIZE {
                return Err(TransportError::InvalidBufferSize { requested: self.send_buffer_size, max: MAX_BUFFER_SIZE });
            }

            let bind_addr = if self.bind_all {
                // placeholder port 0; caller's `bind_with_config` will set port
                Some(SocketAddr::new(Ipv4Addr::new(0, 0, 0, 0).into(), 0))
            } else {
                self.bind_addr
            };

            Ok(TransportConfig {
                bind_addr,
                recv_buffer_size: self.recv_buffer_size,
                send_buffer_size: self.send_buffer_size,
                reuse_port: self.reuse_port,
            })
        }
    }

    /// Binds to the UDP port with configurable and safer defaults.
    /// Backwards-compatible `bind(port)` keeps a single-argument convenience wrapper.
    pub fn bind(port: u16) -> Result<Self> {
        // Allow an operator to explicitly opt-in to binding on all interfaces via env.
        // e.g. AINFTP_BIND_ALL=1
        let mut cfg = TransportConfig::default();
        if env::var("AINFTP_BIND_ALL").ok().as_deref() == Some("1") {
            cfg.bind_addr = Some(SocketAddr::new(Ipv4Addr::new(0, 0, 0, 0).into(), port));
        }
        Self::bind_with_config(port, cfg)
    }

    /// Bind with explicit configuration.
    pub fn bind_with_config(port: u16, cfg: TransportConfig) -> Result<Self> {
        // Validate privileged port usage
        if port != 0 && port < 1024 {
            // If not running as root, refuse to bind privileged ports.
            #[cfg(unix)]
            let euid = unsafe { libc::geteuid() };
            #[cfg(not(unix))]
            let euid = 0;
            if euid != 0 {
                bail!("binding to privileged port <1024 requires  privileges");
            }
        }

        let addr = cfg.bind_addr.unwrap_or_else(|| SocketAddr::new(Ipv4Addr::new(127, 0, 0, 1).into(), port));

        // 1. Create UDP Socket
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))
            .context("creating UDP socket")?;

        // 2. OPTIMIZATION: SO_REUSEPORT (optional)
        if cfg.reuse_port {
            if let Err(e) = socket.set_reuse_port(true) {
                warn!("SO_REUSEPORT not available or failed: {:#}", e);
            }
        }

        // 3. OPTIMIZATION: Increase Kernel Buffer Size (best-effort)
        const MAX_BUFFER_SIZE: usize = 16 * 1024 * 1024; // cap to avoid runaway allocations
        let rcv = std::cmp::min(cfg.recv_buffer_size, MAX_BUFFER_SIZE);
        let snd = std::cmp::min(cfg.send_buffer_size, MAX_BUFFER_SIZE);

        if let Err(e) = socket.set_recv_buffer_size(rcv) {
            warn!("set_recv_buffer_size({}) failed: {:#}", rcv, e);
        }
        if let Err(e) = socket.set_send_buffer_size(snd) {
            warn!("set_send_buffer_size({}) failed: {:#}", snd, e);
        }

        // 3b. Verify effective sizes on Unix (best-effort helper)
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            if let Ok(eff) = unix::get_sockopt_size(socket.as_raw_fd(), libc::SO_RCVBUF) {
                if eff < rcv {
                    warn!("requested recv buffer {} but effective is {}", rcv, eff);
                }
            }
            if let Ok(eff) = unix::get_sockopt_size(socket.as_raw_fd(), libc::SO_SNDBUF) {
                if eff < snd {
                    warn!("requested send buffer {} but effective is {}", snd, eff);
                }
            }
        }

        // 4. Bind
        socket.bind(&addr.into()).with_context(|| format!("binding to {}", addr))?;

        info!("Transport bound to {} (requested rcv/snd {} / {} bytes)", addr, rcv, snd);

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

// Platform-specific helpers
#[cfg(unix)]
mod unix {
    use std::io;

    pub fn get_sockopt_size(fd: i32, opt: libc::c_int) -> io::Result<usize> {
        unsafe {
            let mut val: libc::c_int = 0;
            let mut len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
            let res = libc::getsockopt(fd, libc::SOL_SOCKET, opt, &mut val as *mut _ as *mut _, &mut len);
            if res == 0 {
                Ok(val as usize)
            } else {
                Err(io::Error::last_os_error())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{SocketAddr, Ipv4Addr};

    #[test]
    fn builder_defaults_and_overrides() {
        let b = TransportConfig::builder().recv_buffer(1024).send_buffer(2048).reuse_port(false).build().unwrap();
        assert_eq!(b.recv_buffer_size, 1024);
        assert_eq!(b.send_buffer_size, 2048);
        assert_eq!(b.reuse_port, false);
    }

    #[test]
    fn builder_bind_all_sets_placeholder() {
        let b = TransportConfig::builder().bind_all(true).build().unwrap();
        assert!(b.bind_addr.is_some());
        if let Some(a) = b.bind_addr {
            // placeholder port 0 from builder
            assert_eq!(a.ip(), Ipv4Addr::new(0,0,0,0));
        }
    }

    #[test]
    fn builder_rejects_too_large() {
        let big = 32 * 1024 * 1024;
        let res = TransportConfig::builder().recv_buffer(big).build();
        assert!(res.is_err());
    }
}