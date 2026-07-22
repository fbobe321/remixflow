# RemixFlow — iOS app (skeleton)

An on-device iOS app that runs ACE-Step v1.5 via **MLX-Swift**, using the
components ported and parity-proven in [`../mobile/mlx/`](../mobile/mlx).

> **Status: skeleton.** The app shell (SwiftUI UI, audio engine, pipeline
> orchestration) and the MLX-Swift model code are here. The model files are
> direct translations of the validated Python (VAE, DiT, Qwen3, SDEdit are full;
> the condition encoder ships a runnable *neutral* path + a TODO for the full
> prompt/lyric/timbre conditioning). None of it has been compiled — Swift/Xcode
> needs a Mac. Treat every `*.swift` as reviewed-not-run until it builds on the M1.

## Layout

```
ios/
  RemixFlowKit/                 Swift package (the on-device engine)
    Package.swift               depends on ml-explore/mlx-swift
    Sources/RemixFlowKit/
      MLXOps.swift              rmsNorm, RoPE, snake, SwiGLU, attention
      WeightStore.swift         load safetensors + 4-bit group quantize
      OobleckVAE.swift          encode + decode  (full, from *_mlx.py)
      DiT.swift                 4B transformer   (full, from dit_mlx.py)
      Qwen3TextEncoder.swift    text encoder     (full, from qwen3_mlx.py)
      ConditionEncoder.swift    full forward + neutral fallback (from condenc_mlx.py)
      Pipeline.swift            SDEdit loop + ACEStepPipeline (VAE→DiT→VAE)
  App/                          SwiftUI app (add to an Xcode iOS target)
    RemixFlowApp.swift, ContentView.swift, AudioEngine.swift
```

## Build on the Mac (M1 Air)

1. **Xcode 15+**, create an **iOS App** target (SwiftUI, iOS 17+).
2. Add the package: *File ▸ Add Package Dependencies* → this repo's
   `ios/RemixFlowKit` (or point at `ml-explore/mlx-swift` and this Kit).
3. Add the `App/*.swift` files to the app target; set `RemixFlowApp` as `@main`.
4. **Weights**: download the model
   (`fbobe3/acestep-v15-xl-turbo-diffusers-mirror`), keep the HF folder layout
   (`transformer/ vae/ text_encoder/ condition_encoder/`), and load with
   `WeightStore(directory:)`. Call `store.quantizeLinears(prefix: "transformer/")`
   and the encoders (4-bit, group 32) — the VAE stays fp16. ≈ 3 GB → needs an
   8 GB device (15 Pro / 16). On the 16 GB M1 it fits easily for dev.
5. Instantiate `ACEStepPipeline(store:)` and set it as the `AppModel.generator`.

## Verify parity on-device first

Before trusting generation, confirm each component matches the reference using the
fixtures in `../mobile/mlx/fixtures/` (regenerate with the `export_*.py` scripts):
port each `*_mlx.py`'s `main()` parity check into a small Swift test that loads
`*_input` and compares to `*_output` (tolerance ~1e-3 in fp16). This catches any
MLX-Swift API/layout differences from the Python MLX ports.

## What's wired vs TODO
- ✅ UI (steering sliders, prompt/lyrics, Living toggle), AVAudioEngine playback + gapless enqueue
- ✅ VAE encode/decode, DiT, Qwen3, **ConditionEncoder (full)**, SDEdit loop, 4-bit quantize
- ✅ **Audio decode** (`AudioIO.load`: AVAudioFile → 48 kHz stereo `[2,N]` MLXArray) + WAV write
- ✅ **Tokenizer** (`RFTokenizer` via swift-transformers) → full text/lyric/timbre
  conditioning wired in `ACEStepPipeline` (falls back to neutral if no tokenizer)
- ✅ End-to-end `ACEStepPipeline.generate` (encode → condition → SDEdit(DiT) → decode)
- ⬜ Living Mode loop (generate-ahead + `AudioEngine.enqueue`) on-device
- ⬜ Swap quantize-dequantize for MLX packed `quantizedMatmul` (memory win)
- ⬜ Compile + per-component parity + perf pass on the M1
