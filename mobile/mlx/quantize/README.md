# Quantization — quality measurement

Can ACE-Step v1.5 run at 4-bit on-device without wrecking audio? **Yes.**

MLX's k-bit matmul is numerically equivalent to matmul with dequantized weights,
so the quality can be measured here (no Mac) by quantize→dequantize'ing the DiT's
linear weights and running the real pipeline.

## Result — decoded AUDIO vs fp16 baseline (mel-spectrogram correlation)

| Precision | Audio corr | Single-step velocity SNR | Verdict |
|-----------|-----------:|--------------------------:|---------|
| 8-bit     | **0.9998** | 26.4 dB | perfect |
| **4-bit (group 32)** | **0.9936** | 5.3 dB | **✅ ship it** |
| 4-bit (group 64) | 0.9898 | 3.8 dB | good |
| 3-bit     | (not run) | 2.2 dB | too lossy |

## The key finding

Single-step **velocity SNR is a misleading proxy** — it looks alarming at 4-bit
(3–5 dB), and a mixed-precision sweep (keeping attention or `down_proj` at 8-bit)
barely moved it. But the **decoded audio is 99.4% correlated at 4-bit** anyway,
because RemixFlow's **SDEdit re-anchors to the source every generation** — the
denoise trajectory starts from noised *source* latents, so per-step weight-quant
error doesn't accumulate into audible drift. What matters is the audio, and it
holds up.

## Recommendation

- **DiT + condition/text encoders → 4-bit, group size 32.** Keep the **VAE fp16**
  (quality-critical, only 0.34 GB). Footprint ≈ **3 GB** → fits 8 GB iPhones.
- No mixed precision needed for the DiT — uniform 4-bit gs32 is enough.
- Re-run this test at high `variation_amount` (more regeneration, less
  re-anchoring) before shipping a "reimagine" mode; identity-preserving variation
  (the common case) is safe.

## Files
- `measure_quant.py` — velocity SNR at 8/4/3-bit (fast, but a weak proxy)
- `sweep_quant.py` — mixed-precision SNR sweep
- `audio_test.py` — **the real test**: full pipeline fp16 vs quantized, mel-corr
