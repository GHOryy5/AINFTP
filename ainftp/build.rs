//! ainftp Build Script
//! 
//! 1. Finds the compiled eBPF binary from xtask.
//! 2. (Optional) Compiles .cu files to .ptx if nvcc exists.
//! 3. Tells cargo where to find them.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // 1. Locate the eBPF binary
    // The xtask script puts it here: target/bpfel-unknown-none/release/ainftp
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let target_triple = "bpfel-unknown-none"; // Adjust if you're on ARM
    let profile = env::var("PROFILE").unwrap_or_else(|_| "release".to_string());
    
    let bpf_path = PathBuf::from("target")
        .join(&target_triple)
        .join(&profile)
        .join("ainftp");

    if !bpf_path.exists() {
        panic!(
            "eBPF binary not found at {:?}. Run `cargo xtask build-bpf` first.",
            bpf_path
        );
    }

    // Copy to OUT_DIR so include_bytes_aligned! can find it
    let dest = out_dir.join("bpf_program");
    fs::copy(&bpf_path, &dest)?;

    // Tell cargo to rerun this script if the binary changes
    println!("cargo:rerun-if-changed={}", bpf_path.display());
    
    // 2. CUDA Stub
    // In a real setup, you'd run `nvcc` here. 
    // For now, we assume a .ptx file exists in `src/cuda/` or we skip it.
    
    Ok(())
}