use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser, Debug)]
struct Cli {
    #[clap(subcommand)]
    cmd: Commands,
    #[clap(long, short)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Builds eBPF module (Release)
    Build,
    /// Runs full suite of environment checks
    Audit,
    /// Launches the runtime (Root)
    Run,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Set global log level based on verbose flag
    if !cli.verbose {
        std::env::set_var("RUST_LOG", "warn");
    }

    match cli.cmd {
        Commands::Build => cmd_build().await,
        Commands::Audit => cmd_audit(),
        Commands::Run => cmd_run().await,
    }
}

//  1. AUDIT: HOST CHECKS 

fn cmd_audit() -> Result<()> {
    let spinner = indicatif::ProgressBar::new_spinner();
    spinner.enable_steady_tick();
    spinner.set_message("Checking Host Environment...");

    // 1. Kernel Version (XDP requires 5.8+)
    let output = Command::new("uname")
        .args(["-r"])
        .output()?;
    let raw_ver = String::from_utf8_lossy(output.stdout);
    let major: u32 = raw_ver.split('.')
        .next()
        .unwrap_or("0")
        .parse()?;

    if major < 5 || (major == 5 && raw_ver.contains("4.")) {
        bail!("Kernel too old. XDP needs 5.8+");
    }

    // 2. LLVM (Clang 12+)
    let llvm = Command::new("llc").arg("--version").output()?;
    if !String::from_utf8_lossy(llvm.stdout).contains("12") {
        bail!("LLVM 12 not found. Install clang-12.");
    }

    // 3. Build Tools
    Command::new("cargo-bpf").arg("--version").spawn()?;
    
    spinner.finish_with_message("Host Audit: PASSED");
    Ok(())
}

//  2. BUILD: PARALLEL COMPILATION 

async fn cmd_build() -> Result<()> {
    // Use Aya's builder wrapper
    let mut args = aya_build::BuildArgs::default();
    args.target = "bpfel-unknown-none".to_string();
    args.release = true;

    // Execute Build
    aya_build::build_bpf(args)
        .context("BPF Compilation Failure")?;

    // --- VERIFY ARTIFACTS ---
    let target_dir = PathBuf::from("target/bpfel-unknown-none/release");
    let binary_path = target_dir.join("ainftp");

    if !binary_path.exists() {
        bail!("Build succeeded but binary missing at: {:?}", binary_path);
    }

    // Check binary size (Security: <500kb expected)
    let metadata = std::fs::metadata(&binary_path)?;
    if metadata.len() > 2 * 1024 * 1024 {
        bail!("Binary size suspiciously large (>2MB). Check bloat.");
    }

    println!("[OK] BPF Module Compiled & Verified");
    Ok(())
}

//  3. RUN: EXECUTION 

async fn cmd_run() -> Result<()> {
    let bin = Path::new("target/release/ainftp");
    
    if !bin.exists() {
        bail!("Runtime binary not found. Run 'xtask build' first.");
    }

    // We need root for XDP. Check UID.
    let uid = Command::new("id").arg("-u").output()?;
    let uid_str = String::from_utf8_lossy(uid.stdout);
    if uid_str.trim() != "0" {
        bail!("Root privileges required for XDP attach.");
    }

    println!("[BOOT] Launching ainftp...");
    
    // Inject env vars for performance tuning
    let status = Command::new(&bin)
        .env("RUST_BACKTRACE", "1")
        .env("RUST_LOG", "info,ainftp=debug")
        .status()?;

    if !status.success() {
        bail!("Runtime crashed. Check logs.");
    }

    Ok(())
}