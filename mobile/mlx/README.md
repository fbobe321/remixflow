# RemixFlow on-device (MLX) — Phase 0 spike: VAE decoder

Goal: run ACE-Step v1.5 on an iPhone via **MLX** (Apple Silicon). This is the
first, highest-signal step — porting the **Oobleck VAE decoder** (latent → 48 kHz
stereo waveform) and proving numerical parity, before tackling the 4 B DiT.

## Status — ✅ VAE decoder ported & parity-proven

| Check | Result |
|-------|--------|
| NumPy reimplementation vs PyTorch | **rel err 6.1e-6, corr 1.0000000000** |
| MLX-layout emulation (channels-last + transposed weights) vs PyTorch | **rel err 6.1e-6, corr 1.0000000000** |
| MLX run on Apple Silicon | ⏳ run `vae_decoder_mlx.py` on a Mac |

The decoder is a stack of transposed-conv upsamplers (strides **[10, 6, 4, 4, 2]**,
1920× total → 25 fps latent to 48 kHz), dilated residual units, and **snake**
activations, all weight-normed. Weight-norm is folded at export; snake is
`x + sin²(αx)/β` with log-scale α, β.

## Files

| File | Role | Runs on |
|------|------|---------|
| `vae/export_reference.py` | Load the real VAE, fold weight-norm, dump weights + a parity fixture (input latent + PyTorch output) | Linux/CUDA (done) |
| `vae/oobleck_numpy.py` | Framework-free NumPy spec (the source of truth) | anywhere |
| `vae/_validate_mlx_layout.py` | NumPy emulation of MLX's layout — de-risks the port without a Mac | anywhere |
| `vae/vae_decoder_mlx.py` | **The MLX port** + parity test | **Apple Silicon** |
| `fixtures/` | `vae_decoder.safetensors`, config, `parity_input.npy`, `parity_output.npy` | — |

## Run the MLX parity test (on a Mac / cloud Mac)

```bash
pip install mlx numpy safetensors
cd mobile/mlx/vae
python vae_decoder_mlx.py
# expect: PARITY_PASS  (rel err < 1e-3 vs the PyTorch fixture)
```

No Apple Silicon handy? MLX can't be virtualized on x86 (needs Metal). Options:
**a real M-series Mac**, a **cloud Mac** (AWS `mac2`, MacStadium, Scaleway),
or **GitHub Actions `macos-14/15` runners** (Apple Silicon) for parity CI.

## Notes toward the full on-device model

- **Quantization plan:** DiT + condition/text encoders → **4-bit** (`mlx.nn.quantize`,
  group 32/64), **VAE stays fp16** (quality-critical, only 0.34 GB). Total ≈ **3 GB**
  → fits **8 GB iPhones** (15 Pro / all 16s), not 6 GB.
- **VAE memory:** decode is memory-heavy at the widest layer (128 ch × output length).
  **Tile** to ~4–8 s windows on-device (the pipeline already tiles); RemixFlow's
  Living windows are ~12 s and can be sub-tiled.
- **Next ports:** (1) Qwen3 text encoder — near-turnkey via `mlx_lm.convert -q`;
  (2) the 4 B DiT — standard modern transformer (RMSNorm, QK-norm, SwiGLU, adaLN
  via `scale_shift_table`, cross-attn, dual timestep embed) → hand-port + 4-bit;
  (3) condition encoder; (4) the flow-matching SDEdit loop (trivial).
