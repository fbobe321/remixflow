# RemixFlow on-device (MLX) — running ACE-Step v1.5 on iPhone

Goal: run ACE-Step v1.5 on Apple Silicon via **MLX**. Approach: reimplement each
component framework-free, **prove numerical parity against the real PyTorch model
here**, then translate to MLX (a mechanical step) with a fixture the Mac verifies.

## Status

| Component | Params | NumPy vs PyTorch | MLX (run on Mac) |
|-----------|-------:|------------------|------------------|
| **VAE decoder** (Oobleck) | 0.17 B | ✅ **6.1e-6, corr 1.0** | `vae/vae_decoder_mlx.py` |
| **DiT** (AceStepTransformer1DModel) | **4.17 B** | ✅ **8.8e-6, corr 1.0** | `dit/dit_mlx.py` |
| **Qwen3 text encoder** | 0.60 B | ✅ **3.5e-6, corr 1.0** | `textenc/qwen3_mlx.py` |
| **Condition encoder** | 0.61 B | ✅ **3.4e-7, corr 1.0** (mask exact) | `condenc/condenc_mlx.py` |
| VAE **encoder** (for SDEdit) | (in VAE) | ⬜ | — |
| Flow-matching SDEdit loop | — | ⬜ (trivial glue) | — |

**✅ All 4 weight components ported & parity-proven (5.54 B / 5.54 B params).**
Remaining is non-weight plumbing: the VAE encoder (waveform→latent for SDEdit),
the flow-matching loop, tokenizer, and wiring the full MLX pipeline. See `TODO.md`.

Both ported components' **MLX layout logic is also validated in NumPy** (VAE:
`_validate_mlx_layout.py`; DiT: conv/deconv patchify unit test), so the only thing
left for these two is confirming MLX's own ops on a Mac.

The decoder is a stack of transposed-conv upsamplers (strides **[10, 6, 4, 4, 2]**,
1920× total → 25 fps latent to 48 kHz), dilated residual units, and **snake**
activations, all weight-normed. Weight-norm is folded at export; snake is
`x + sin²(αx)/β` with log-scale α, β.

## Files

| File | Role |
|------|------|
| `vae/oobleck_numpy.py` | VAE decoder — framework-free spec (source of truth) |
| `vae/vae_decoder_mlx.py` | VAE decoder — **MLX port** + parity test |
| `vae/export_reference.py` | Fold weight-norm, dump VAE parity fixture |
| `vae/_validate_mlx_layout.py` | NumPy emulation of MLX conv layout (de-risk without a Mac) |
| `dit/dit_numpy.py` | DiT — framework-free spec (dual timestep, GQA+QK-norm+RoPE, sliding/full attn, AdaLN, SwiGLU, patchify) |
| `dit/dit_mlx.py` | DiT — **MLX port** + parity test |
| `dit/export_reference.py` | Capture a DiT reference forward (seeded synthetic inputs) |
| `textenc/qwen3_numpy.py` | Qwen3 text encoder — framework-free spec (GQA+QK-norm+RoPE+SwiGLU, causal, final norm) |
| `textenc/qwen3_mlx.py` | Qwen3 — **MLX port** + parity test |
| `textenc/export_reference.py` | Capture a Qwen3 `last_hidden_state` reference |
| `condenc/condenc_numpy.py` | Condition encoder spec (text proj + lyric/timbre encoders + pack/unpack) |
| `condenc/condenc_mlx.py` | Condition encoder — **MLX port** + parity test |
| `condenc/export_reference.py` | Capture a condition-encoder reference (synthetic inputs) |
| `fixtures/` | parity fixtures (git-ignored; regenerate via the export scripts) |

DiT weights are loaded straight from the HF snapshot's `transformer/*.safetensors`
(bf16 → fp32 via torch), so nothing large is re-exported.

## Run the MLX parity tests (on a Mac / cloud Mac)

```bash
pip install mlx numpy safetensors torch
# 1. regenerate fixtures once (needs the model; run on any machine with it):
python vae/export_reference.py
python dit/export_reference.py
# 2. parity on Apple Silicon:
python vae/vae_decoder_mlx.py                    # -> PARITY_PASS
python dit/dit_mlx.py /path/to/hf/snapshot       # -> DIT_MLX_PARITY_PASS
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
