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
- ⬜ VAE **encoder** (needed for SDEdit: waveform → latent)
- ⬜ Flow-matching SDEdit loop (pure array math; noise → 8 Euler steps → decode)
- ⬜ Tokenizer (Qwen2TokenizerFast) — bundle or swift-transformers
- ⬜ Wire the full pipeline in MLX (encode → condition → denoise → decode)
- ⬜ Confirm every `*_mlx.py` parity test on the M1 Air

## Quantization (de-risk accuracy)
- ⬜ 4-bit quantize DiT + condition/text encoders (`mlx.nn.quantize`, keep VAE fp16)
- ⬜ Measure quality hit (A/B quantized vs fp16 output) while the PyTorch reference
  is still available here
- ⬜ Mixed precision if needed (keep norms/scale_shift/timestep at fp16)
- ⬜ Confirm ~3 GB footprint fits 8 GB iPhone budget

## iOS app
- ⬜ Swift + MLX app skeleton (load quantized weights, run pipeline)
- ⬜ Port the steering UI (SwiftUI) OR wrap the web UI (PWA/Capacitor)
- ⬜ Streaming/gapless Living playback on-device (AVAudioEngine)
- ⬜ Decide vocal-preservation strategy on mobile (defer / server-side Demucs)
- ⬜ On-device perf pass (tiling for VAE memory; buffer-ahead for Living)

## Server / distribution (alt path)
- ⬜ RemoteGenerator backend (`REMIXFLOW_MODEL_URL`) — thin client → hosted model
  (enables a lightweight phone app talking to a GPU server)

## Housekeeping
- ⬜ **Rotate the 4 tokens** pasted in chat (GitHub, PyPI, HF, Docker)
- ⬜ Deeper Musical DNA + real identity score (sections, genre, emotional dims)
- ⬜ Per-feature morphing (tempo/melody/chords drift independently)
