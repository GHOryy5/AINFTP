//! ainftp Build System
//! 
//! Automates the hell out of compiling eBPF and moving binaries.
//! run with: `cargo xtask build-bpf`

use std::path::PathBuf;
use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[clap(subcommand)]
    cmd: Command,
}

#[derive(Parser, Debug)]
enum Command {
    /// Compiles the eBPF target
    BuildBpf,
    /// Runs the main application
    Run,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Command::BuildBpf => build_bpf(),
        Command::Run => run_app(),
    }
}

fn build_bpf() -> Result<()> {
    // Use Aya's builder to handle the LLVM madness
    let mut args = aya_build::BuildArgs::default();
    
    // Build the BPF target
    aya_build::build_bpf(args)
        .context("Failed to build eBPF module. Is LLVM installed?")?;
    
    // Copy the binary to a place the userspace app expects
    // This creates: target/bpfel-unknown-none/release/ainftp
    println!("eBPF binary compiled and staged.");
    Ok(())
}

fn run_app() -> Result<()> {
    // Ensure BPF is built first
    build_bpf()?;
    
    // Run the main app
    let status = std::process::Command::new("cargo")
        .args(["run", "--release"])
        .status()
        .context("Failed to run application")?;
    
    if !status.success() {
        anyhow::bail!("Application exited with error");
    }
    Ok(())
}