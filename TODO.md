# RemixFlow — TODO / Roadmap

Status legend: ✅ done · 🟡 in progress · ⬜ not started

## Shipped / deployed
- ✅ Phase 1 app (steering UI, FastAPI, DSP + ACE-Step v1.5 backends, async jobs,
  evolution tree, A/B, morph, preference learning, vocal preservation)
- ✅ Phase 2 Living Songs (continuous engine, gapless Web-Audio player, tension
  model, memory, listening-mode presets + user presets, playlists)
- ✅ GitHub: https://github.com/fbobe321/remixflow
- ✅ PyPI: `pip install "remixflow[audio]"` (0.1.1)
- ✅ Docker Hub: `fbobe3/remixflow:latest` (slim) + `Dockerfile.gpu`
- ✅ HF model mirror: `fbobe3/acestep-v15-xl-turbo-diffusers-mirror`

## On-device (MLX) — port ACE-Step v1.5 to Apple Silicon / iPhone
Recipe per component: reimplement framework-free → prove parity vs PyTorch here →
translate to MLX → confirm on Mac (M1 Air, 16 GB, inbound).

- ✅ VAE decoder (Oobleck) — parity 6.1e-6
- ✅ DiT (4.17B transformer) — parity 8.8e-6
- ✅ Qwen3 text encoder (0.60B) — parity 3.5e-6
- ✅ Condition encoder (0.61B) — parity 3.4e-7 (mask exact). **All 4 weight
  components done — 5.54B/5.54B.**
- ✅ VAE **encoder** (SDEdit waveform → latent) — parity 3.9e-6
- ✅ Flow-matching SDEdit loop — 1e-7 vs diffusers scheduler
- ⬜ Tokenizer (Qwen2TokenizerFast) — bundle or swift-transformers
- ⬜ Wire the full pipeline in MLX (encode → condition → denoise → decode) end-to-end
- ⬜ Confirm every `*_mlx.py` parity test on the M1 Air

## Quantization (de-risk accuracy) — ✅ MEASURED, 4-bit viable
- ✅ Measured quality: **4-bit gs32 → decoded audio 0.9936 mel-corr vs fp16**
  (8-bit 0.9998). SDEdit re-anchoring hides per-step quant error. `mobile/mlx/quantize/`.
- ✅ Decision: uniform **4-bit group-32** for DiT + encoders, **VAE fp16** → ~3 GB.
- ⬜ Apply real `mlx.nn.quantize` on the Mac + confirm on-device output matches.
- ⬜ Re-test at high `variation_amount` before a "reimagine" mode ships.

## iOS app
- 🟡 Swift + MLX app — **built** in `ios/` (RemixFlowKit + SwiftUI app).
  MLX-Swift: VAE/DiT/Qwen3/ConditionEncoder/SDEdit **all full**; WeightStore
  (load+4-bit); **AudioIO** (decode/resample→[2,N]@48k + WAV write); **RFTokenizer**
  (swift-transformers); **full conditioning wired** in ACEStepPipeline; AudioEngine;
  ContentView (prompt/lyrics/sliders). **Not compiled yet (needs Mac).**
- ⬜ Compile + run on the M1; port each `*_mlx.py` parity check into a Swift test
- ⬜ Living Mode loop on-device (generate-ahead + AudioEngine.enqueue)
- ⬜ Swap quant-dequant for MLX packed quantizedMatmul (memory win)
- ⬜ Decide vocal-preservation on mobile (defer / server-side Demucs)
- ⬜ On-device perf pass (VAE tiling; buffer-ahead)

## Server / distribution (alt path)
- ⬜ RemoteGenerator backend (`REMIXFLOW_MODEL_URL`) — thin client → hosted model
  (enables a lightweight phone app talking to a GPU server)

## Housekeeping
- ⬜ **Rotate the 4 tokens** pasted in chat (GitHub, PyPI, HF, Docker)
- ⬜ Deeper Musical DNA + real identity score (sections, genre, emotional dims)
- ⬜ Per-feature morphing (tempo/melody/chords drift independently)
