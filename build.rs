//! Portable native CUDA + Tauri v2 build script for `rust-singularity-core-v3`.
//! Supports `CUDA_HOME`, `NVCC`, and `CUDA_ARCH` environment variables, with automatic
//! detection of `/usr/local/cuda-12.9`, `/usr/local/cuda`, or `nvcc` in `PATH`.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn detect_cuda_home() -> PathBuf {
    if let Ok(ch) = env::var("CUDA_HOME") {
        let p = PathBuf::from(ch);
        if p.exists() {
            return p;
        }
    }
    for cand in ["/usr/local/cuda-12.9", "/usr/local/cuda"] {
        let p = PathBuf::from(cand);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("/usr/local/cuda")
}

fn main() {
    println!("cargo:rerun-if-changed=cuda/v3_kernels.cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR missing"));
    let obj_path = out_dir.join("v3_kernels.o");
    let lib_path = out_dir.join("libv3_cuda.a");

    let cuda_home = detect_cuda_home();
    let nvcc = env::var("NVCC").unwrap_or_else(|_| {
        let bin = cuda_home.join("bin").join("nvcc");
        if bin.exists() {
            bin.display().to_string()
        } else {
            "nvcc".to_string()
        }
    });
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_120".to_string());

    let status = Command::new(&nvcc)
        .args([
            "-O3",
            &format!("-arch={cuda_arch}"),
            "--fmad=true",
            "-std=c++17",
            "-Xcompiler",
            "-fPIC,-Wall,-O3",
            "-c",
            "cuda/v3_kernels.cu",
            "-o",
        ])
        .arg(&obj_path)
        .status()
        .unwrap_or_else(|e| panic!("Failed to invoke nvcc ({nvcc}): {e}. Set CUDA_HOME or NVCC."));

    if !status.success() {
        panic!("CUDA compilation ({cuda_arch}) failed with status {:?}. Override GPU arch via CUDA_ARCH=sm_86 (or your GPU compute capability).", status.code());
    }

    let ar_status = Command::new("ar")
        .arg("crs")
        .arg(&lib_path)
        .arg(&obj_path)
        .status()
        .expect("Failed to run ar crs libv3_cuda.a");

    if !ar_status.success() {
        panic!("Packaging libv3_cuda.a failed with code {:?}", ar_status.code());
    }

    let cuda_lib64 = cuda_home.join("lib64");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib64.display());
    println!("cargo:rustc-link-search=native=/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib64.display());
    println!("cargo:rustc-link-lib=static=v3_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=sqlite3");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    tauri_build::build();
}
