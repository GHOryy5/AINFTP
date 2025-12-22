# ainftp (v1) // the network reflex

**Standard Linux networking (TCP/IP) was built 40 years ago for emails. It wasn't built to stream gigabytes of gradients for AGI.**

When you train distributed models , the kernel is the bottleneck. Every time a packet hits the NIC, the CPU wakes up, context switches, and runs a mountain of legacy code. For a GPU waiting on data, that latency is an eternity.

**ainftp** creates a "reflex" in the network stack using **Rust + eBPF**. We don't ask the OS for permission. We grab the data at the driver level and move it.

---

## what it actually does
We use **XDP (eXpress Data Path)** to hijack packets the microsecond they hit the hardware.

1.  **Kernel Bypass:** Identifies AI tensors/gradients and strips them off the wire before the Linux kernel even knows they exist.
2.  **Virtual RDMA:** Uses shared BPF maps to bridge network data directly into userspace/GPU memory. Zero copy. Zero context switches.
3.  **In   Network Aggregation:** Lays the pipework to merge gradients *inside* the network stack.

##  why this exists
I'm building this because "Internet Jitter" is killing distributed training.

If you have 100 nodes, **one** slow node (a straggler) pauses the entire cluster. The standard network stack is too dumb to fix this.

**ainftp** tags training traffic as "Critical" at the hardware level. If a node lags, we detect it in microseconds at Ring 0 and handle it.

## stack
*   **Language:** RUST
*   **Kernel:** eBPF / XDP via `Aya`.
*   **Compute:** CUDA via `cudarc` for direct memory injection.

## structure
*   `ainftp-ebpf`: **The Reflex.** Code injected into the kernel.
*   `ainftp-common`: **The Synapse.** Shared memory maps.
*   `ainftp`: **The Brain.** User-space controller.
