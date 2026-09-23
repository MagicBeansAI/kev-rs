//! Fused Metal kernel for the Gated DeltaNet recurrence, called through the
//! raw mlx-sys C API (mlx-rs 0.32 does not expose `mx.fast.metal_kernel`).
//!
//! The Metal source is the scalar-gating, unmasked variant of
//! `_make_gated_delta_kernel` from mlx-lm 0.31.3 (`mlx_lm/models/gated_delta.py`,
//! MIT License, Copyright © Apple Inc.), reproduced verbatim so the Rust MLX
//! backend runs the same one-launch-per-layer prefill as the Python baseline.
//! One thread simdgroup (32 lanes) owns Dk; the grid covers (32, Dv, B*Hv);
//! the sequential loop over T lives inside the kernel.

use crate::error::{KevError, Result};
use mlx_rs::Array;
use std::ffi::CString;

const SOURCE: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    // q, k: [B, T, Hk, Dk]
    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

    // v, y: [B, T, Hv, Dv]
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    // state_in, state_out: [B, Hv, Dv, Dk]
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }

    // g: [B, T, Hv]
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      if (true) {
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] * g_[hv_idx];
          kv_mem += state[i] * k_[s_idx];
        }
        kv_mem = simd_sum(kv_mem);

        auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] + k_[s_idx] * delta;
          out += state[i] * q_[s_idx];
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) {
          y[dv_idx] = static_cast<InT>(out);
        }
      } else {
        y[dv_idx] = static_cast<InT>(0);
      }
      // Increment data pointers to next time step
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
"#;

pub struct GdnKernel {
    kernel: mlx_sys::mlx_fast_metal_kernel,
}

// The kernel handle is an immutable compiled-program reference.
unsafe impl Send for GdnKernel {}

impl Drop for GdnKernel {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_free(self.kernel);
        }
    }
}

fn ffi(status: std::os::raw::c_int, what: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(KevError::Inference(format!(
            "gdn kernel: {what} failed ({status})"
        )))
    }
}

impl GdnKernel {
    pub fn new() -> Result<Self> {
        let name = CString::new("gated_delta_step").unwrap();
        let source = CString::new(SOURCE).unwrap();
        let header = CString::new("").unwrap();
        unsafe {
            let input_names = mlx_sys::mlx_vector_string_new();
            for input in ["q", "k", "v", "g", "beta", "state_in", "T"] {
                let s = CString::new(input).unwrap();
                ffi(
                    mlx_sys::mlx_vector_string_append_value(input_names, s.as_ptr()),
                    "input name",
                )?;
            }
            let output_names = mlx_sys::mlx_vector_string_new();
            for output in ["y", "state_out"] {
                let s = CString::new(output).unwrap();
                ffi(
                    mlx_sys::mlx_vector_string_append_value(output_names, s.as_ptr()),
                    "output name",
                )?;
            }
            let kernel = mlx_sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                input_names,
                output_names,
                source.as_ptr(),
                header.as_ptr(),
                true, // ensure_row_contiguous
                false,
            );
            mlx_sys::mlx_vector_string_free(input_names);
            mlx_sys::mlx_vector_string_free(output_names);
            if kernel.ctx.is_null() {
                return Err(KevError::Load(
                    "gated delta metal kernel creation failed".into(),
                ));
            }
            Ok(Self { kernel })
        }
    }

    /// q,k: [B,T,Hk,Dk] (InT); v: [B,T,Hv,Dv] (InT); g,beta: [B,T,Hv];
    /// state: [B,Hv,Dv,Dk] (f32). Returns (y [B,T,Hv,Dv] InT, state_out f32).
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state: &Array,
    ) -> Result<(Array, Array)> {
        let k_shape = k.shape();
        let (b, t, _hk_dim, _) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
        let hk = k_shape[2];
        let dk = k_shape[3];
        let v_shape = v.shape();
        let hv = v_shape[2];
        let dv = v_shape[3];
        if dk % 32 != 0 {
            return Err(KevError::Inference(format!(
                "gdn kernel requires Dk divisible by 32, got {dk}"
            )));
        }

        unsafe {
            let config = mlx_sys::mlx_fast_metal_kernel_config_new();
            let y_shape: [i32; 4] = [b, t, hv, dv];
            let in_dtype = mlx_sys::mlx_array_dtype(q.as_ptr());
            let state_dtype = mlx_sys::mlx_array_dtype(state.as_ptr());
            ffi(
                mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                    config,
                    y_shape.as_ptr(),
                    4,
                    in_dtype,
                ),
                "output y",
            )?;
            let state_shape: [i32; 4] = [b, hv, dv, dk];
            ffi(
                mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                    config,
                    state_shape.as_ptr(),
                    4,
                    state_dtype,
                ),
                "output state",
            )?;
            ffi(
                mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 32, dv, b * hv),
                "grid",
            )?;
            ffi(
                mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1),
                "threadgroup",
            )?;
            for (tname, dtype) in [("InT", in_dtype), ("StT", state_dtype)] {
                let s = CString::new(tname).unwrap();
                ffi(
                    mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                        config,
                        s.as_ptr(),
                        dtype,
                    ),
                    "template dtype",
                )?;
            }
            for (tname, value) in [("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv)] {
                let s = CString::new(tname).unwrap();
                ffi(
                    mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                        config,
                        s.as_ptr(),
                        value,
                    ),
                    "template int",
                )?;
            }

            let t_scalar = mlx_sys::mlx_array_new_int(t);
            let inputs = mlx_sys::mlx_vector_array_new();
            for array in [q, k, v, g, beta, state] {
                ffi(
                    mlx_sys::mlx_vector_array_append_value(inputs, array.as_ptr()),
                    "input",
                )?;
            }
            ffi(
                mlx_sys::mlx_vector_array_append_value(inputs, t_scalar),
                "input T",
            )?;

            let mut outputs = mlx_sys::mlx_vector_array_new();
            let stream = mlx_sys::mlx_default_gpu_stream_new();
            let status = mlx_sys::mlx_fast_metal_kernel_apply(
                &mut outputs,
                self.kernel,
                inputs,
                config,
                stream,
            );
            mlx_sys::mlx_stream_free(stream);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_array_free(t_scalar);
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            if status != 0 {
                mlx_sys::mlx_vector_array_free(outputs);
                return Err(KevError::Inference(
                    "gated delta kernel apply failed".into(),
                ));
            }

            let mut y_raw = mlx_sys::mlx_array_new();
            ffi(
                mlx_sys::mlx_vector_array_get(&mut y_raw, outputs, 0),
                "get y",
            )?;
            let mut state_raw = mlx_sys::mlx_array_new();
            ffi(
                mlx_sys::mlx_vector_array_get(&mut state_raw, outputs, 1),
                "get state",
            )?;
            mlx_sys::mlx_vector_array_free(outputs);
            Ok((Array::from_ptr(y_raw), Array::from_ptr(state_raw)))
        }
    }
}
