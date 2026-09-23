//! Link check: mlx-rs 0.32.0 against laya-rs's patched mlx-sys, pulled as a
//! git dependency (the exact pin systemone carries). Runs one op on the GPU
//! stream and one on the CPU stream, prints versions and the metallib mode.

use anyhow::Result;
use mlx_rs::{
    ops::{matmul, mean},
    transforms::eval,
    Array, Dtype, StreamOrDevice,
};

fn main() -> Result<()> {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]);

    // Default (GPU) stream.
    let gpu = matmul(&a, &b)?;
    eval([&gpu])?;

    // CPU stream, as the upstream fp32 LoRA merge requires.
    let cpu_stream = StreamOrDevice::cpu();
    let cpu = mlx_rs::ops::matmul_device(&a, &b, &cpu_stream)?;
    eval([&cpu])?;

    let diff = mean(&mlx_rs::ops::abs(&gpu.subtract(&cpu)?)?, false)?;
    eval([&diff])?;

    println!(
        "{{\"ok\": true, \"gpu_dtype\": \"{:?}\", \"gpu_vs_cpu_mean_abs\": {}, \"metal_jit\": \"{}\", \"deployment_target\": \"{}\"}}",
        Dtype::Float32,
        diff.item::<f32>(),
        std::env::var("MLX_RS_METAL_JIT").unwrap_or_default(),
        std::env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_default(),
    );
    Ok(())
}
