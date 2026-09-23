//! Early linking check (K1): laya-core's MLX runtime, kev's mlx-rs usage and
//! llama.cpp (llama-cpp-2, systemone's exact version) in one binary, sharing
//! one patched mlx-sys. Initializes each stack and runs a trivial op.

use anyhow::Result;
use mlx_rs::{ops::indexing::IndexOp, ops::matmul, transforms::eval, Array};

fn main() -> Result<()> {
    // kev MLX path: one GPU op through the shared patched mlx-sys.
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let product = matmul(&a, &a)?;
    eval([&product])?;

    // laya-core linked and its MLX backend compiled in.
    let laya_types = std::any::type_name::<laya_core::LoadOptions>();

    // llama.cpp backend initializes in the same process.
    let backend = llama_cpp_2::llama_backend::LlamaBackend::init()?;
    drop(backend);

    println!(
        "{{\"ok\": true, \"mlx_matmul_00\": {}, \"laya_core\": \"{laya_types}\", \"llama_cpp\": \"initialized\"}}",
        product.index((0, 0)).item::<f32>()
    );
    Ok(())
}
