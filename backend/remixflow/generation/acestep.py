"""ACE-Step v1.5 generation backend (via the diffusers ``AceStepPipeline``).

RemixFlow's "keep the identity, vary the style" maps onto ACE-Step's
``task_type="cover"`` audio-to-audio path:

  parent audio            -> src_audio + reference_audio (48 kHz stereo tensors)
  variation_amount        -> audio_cover_strength (+ identity locks lower it)
  genre/tone/energy/vocal -> a synthesized text `prompt`
  detected tempo/key      -> bpm / keyscale hints (when available)
  seed                    -> torch.Generator (reproducible)

The pipeline is a lazy GPU singleton, loaded once and reused. If torch /
diffusers / weights are unavailable the backend reports ``available = False`` so
the API still boots and the DSP backend serves as a fallback.

Env overrides:
  ACESTEP_MODEL           HF repo id (default acestep-v15-xl-turbo-diffusers)
  ACESTEP_DEVICE          cuda | cuda:1 | cpu   (default cuda)
  ACESTEP_DTYPE           bfloat16 | float16 | float32 (default bfloat16)
  ACESTEP_STEPS           num_inference_steps (default: model default, 8 turbo)
  ACESTEP_GUIDANCE        guidance_scale (base/sft only; turbo ignores it)
  ACESTEP_STRENGTH_INVERT "1" if lower audio_cover_strength should mean MORE change
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

from ..audio.io import Clip
from ..models import Steering
from ..params import CONTROLS_BY_KEY
from .base import GenerationResult, Generator

DEFAULT_MODEL = "ACE-Step/acestep-v15-xl-turbo-diffusers"


def _probe() -> bool:
    """True if torch + the diffusers ACE-Step pipeline are importable."""
    try:
        import torch  # noqa: F401
        from diffusers import AceStepPipeline  # noqa: F401
        return True
    except Exception:
        return False


class AceStepGenerator(Generator):
    name = "ace-step"
    description = "ACE-Step v1.5 diffusion (audio2audio cover). Real generative model."

    def __init__(self) -> None:
        self.available = _probe()
        self._pipe = None
        self._lock = threading.Lock()

    # --- lazy pipeline singleton ------------------------------------------

    def _pipeline(self):
        if self._pipe is not None:
            return self._pipe
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            import torch
            from diffusers import AceStepPipeline

            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }.get(os.environ.get("ACESTEP_DTYPE", "bfloat16"), torch.bfloat16)
            model = os.environ.get("ACESTEP_MODEL", DEFAULT_MODEL)
            device = os.environ.get("ACESTEP_DEVICE", "cuda")

            pipe = AceStepPipeline.from_pretrained(model, torch_dtype=dtype)
            pipe = pipe.to(device)
            self._pipe = pipe
            self._device = device
            return self._pipe

    # --- steering -> ACE-Step params --------------------------------------

    def build_prompt(self, steering: Steering, instrumental: bool = False) -> str:
        """Turn the slider state into a natural-language style prompt.

        ``instrumental=True`` (used when varying a vocals-removed backing track)
        tells the model to keep it instrumental, so it doesn't hallucinate vocal
        sounds under the preserved real vocals.
        """
        c = steering.controls

        def val(key: str) -> float:
            return float(c.get(key, 0.0) or 0.0)

        parts: list[str] = []
        for key in ("rock", "blues", "jazz", "electronic", "acoustic"):
            v = val(key)
            if v > 0.35:
                label = CONTROLS_BY_KEY[key].label.replace(" Influence", "").lower()
                parts.append(("heavy " if v > 0.7 else "") + label)

        bipolar = {
            "energy": ("mellow", "energetic"),
            "brightness": ("dark", "bright"),
            "warmth": ("cold", "warm"),
            "bass": ("light bass", "heavy bass"),
            "emotional_tone": ("melancholic", "uplifting"),
            "complexity": ("minimal", "intricate"),
            "instrument_density": ("sparse arrangement", "dense arrangement"),
            "groove": ("loose groove", "tight groove"),
        }
        for key, (lo, hi) in bipolar.items():
            v = val(key)
            if v > 0.4:
                parts.append(hi)
            elif v < -0.4:
                parts.append(lo)

        if val("swing") > 0.4:
            parts.append("swung rhythm")
        if val("chorus_strength") > 0.4:
            parts.append("anthemic chorus")

        focus = c.get("instrument_focus")
        if isinstance(focus, list) and focus:
            parts.append("prominent " + ", ".join(focus))

        vocal = c.get("vocal_style")
        if isinstance(vocal, str) and vocal and vocal != "original":
            parts.append(f"{vocal} vocals")

        if instrumental:
            parts = ["instrumental", "no vocals"] + parts
        base = ", ".join(parts) if parts else "faithful cover, same style and mood"
        return base

    def _strength(self, steering: Steering) -> float:
        """0..1 SDEdit noise level from variation amount, reduced by identity locks.

        This is the fraction of the diffusion trajectory we re-run: high = start
        near pure noise (reimagined), low = start near the clean source (nearly
        identical). Each identity lock pulls it back toward the source.
        """
        amt = float(steering.controls.get("variation_amount", 0.2) or 0.0)
        strength = amt - 0.05 * len(steering.locks)
        return round(min(0.98, max(0.08, strength)), 3)

    # --- generation --------------------------------------------------------

    def generate(
        self,
        parent: Clip,
        steering: Steering,
        *,
        original: Optional[Clip] = None,
        seed: Optional[int] = None,
        instrumental: bool = False,
    ) -> GenerationResult:
        if not self.available:
            raise RuntimeError(
                "ACE-Step backend unavailable (torch/diffusers/weights missing)."
            )
        s = steering.normalized()
        pipe = self._pipeline()
        sr = int(getattr(pipe, "sample_rate", 48000))
        strength = self._strength(s)
        prompt = self.build_prompt(s, instrumental=instrumental)

        preserve_vocals = self._should_preserve_vocals(s)
        if preserve_vocals:
            # Identity: keep the ORIGINAL vocals. Split them off, vary only the
            # instrumental, then remix the real vocals back on top (tempo/length
            # preserved by SDEdit, so they stay aligned).
            vocals, instrumental = self._separate(parent)  # [2,N] @ parent.sr
            instr = self._to_model_tensor(
                Clip(samples=instrumental, sample_rate=parent.sample_rate), sr)
            samples, note = self._sdedit_audio(pipe, instr, strength, prompt, seed, sr)
            samples = self._remix_vocals(samples, sr, vocals, parent.sample_rate)
            note = note + " · original vocals preserved"
        else:
            src = self._to_model_tensor(parent, sr)  # [2, N] float32 CPU tensor
            samples, note = self._sdedit_audio(pipe, src, strength, prompt, seed, sr)

        if samples.ndim == 1:
            samples = samples[np.newaxis, :]
        # Final peak-normalize to -0.5 dBFS. Crossfade overlaps, resampling, and the
        # vocal remix can push summed peaks over 1.0 (clipping) even though each
        # chunk is normalized internally; this guarantees a clean output level.
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 1e-6:
            samples = samples * (0.944 / peak)  # 0.944 ≈ -0.5 dBFS
        result_clip = Clip(samples=samples.astype(np.float32), sample_rate=sr)
        return GenerationResult(clip=result_clip, generator=self.name, note=note)

    def _sdedit_audio(self, pipe, src, strength: float, prompt: str,
                      seed: Optional[int], sr: int):
        """Run SDEdit over an audio tensor [2,N], chunking long clips. -> (np[2,N], note)."""
        chunk_sec = float(os.environ.get("ACESTEP_CHUNK_SEC", "30"))
        overlap_sec = float(os.environ.get("ACESTEP_OVERLAP_SEC", "2"))
        total = src.shape[-1]
        chunked = total > int((chunk_sec + overlap_sec) * sr)

        if not chunked:
            samples, n_steps = self._sdedit_segment(pipe, src, strength, prompt, seed)
            return samples, f"sdedit strength={strength} steps={n_steps} prompt='{prompt[:48]}'"

        hop = int((chunk_sec - overlap_sec) * sr)
        win = int(chunk_sec * sr)
        ov = int(overlap_sec * sr)
        pieces: list[np.ndarray] = []
        n_steps = 0
        for i, st in enumerate(range(0, total, hop)):
            seg = src[:, st:min(st + win, total)]
            if seg.shape[-1] < int(0.5 * sr):  # skip a tiny trailing sliver
                continue
            seg_seed = None if seed is None else int(seed) + i
            seg_out, n_steps = self._sdedit_segment(pipe, seg, strength, prompt, seg_seed)
            pieces.append(seg_out)
        samples = self._crossfade_concat(pieces, ov)
        return samples, (f"sdedit(chunked x{len(pieces)}) strength={strength} "
                         f"steps={n_steps} prompt='{prompt[:40]}'")

    # --- vocal preservation (Demucs stem separation) ----------------------

    #: Locking either of these identity elements triggers vocal preservation.
    VOCAL_LOCKS = frozenset({"lyrics", "vocal_phrasing"})

    def _should_preserve_vocals(self, steering: Steering) -> bool:
        if os.environ.get("ACESTEP_PRESERVE_VOCALS", "1") != "1":
            return False
        if not (self.VOCAL_LOCKS & set(steering.locks)):
            return False
        return self._separator() is not None

    def _separator(self):
        """Lazy Demucs separator singleton, or None if Demucs isn't installed.

        Device is resolved from the environment (not the pipeline) so this works
        regardless of init order; failures are not cached, so a transient error
        doesn't permanently disable the feature.
        """
        if getattr(self, "_sep", None) is not None:
            return self._sep
        try:
            from demucs.api import Separator
            device = os.environ.get("ACESTEP_DEVICE", "cuda")
            self._sep = Separator(model="htdemucs", device=device)
        except Exception:
            self._sep = None
        return self._sep

    def _separate(self, clip: Clip):
        """Split a clip into (vocals, instrumental), both numpy [2, N] @ clip.sr."""
        import torch

        sep = self._separator()
        wav = torch.from_numpy(clip.samples.astype(np.float32))
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        _, stems = sep.separate_tensor(wav, clip.sample_rate)
        vocals = stems["vocals"].cpu().numpy()
        instrumental = sum(stems[k] for k in ("drums", "bass", "other")).cpu().numpy()
        return vocals, instrumental

    @staticmethod
    def _remix_vocals(instr, instr_sr: int, vocals, vocals_sr: int):
        """Overlay original vocals onto the varied instrumental (both stereo)."""
        v_gain = float(os.environ.get("ACESTEP_VOCAL_GAIN", "1.0"))
        i_gain = float(os.environ.get("ACESTEP_INSTR_GAIN", "1.0"))
        if vocals_sr != instr_sr:
            import librosa
            vocals = np.stack([
                librosa.resample(vocals[c], orig_sr=vocals_sr, target_sr=instr_sr)
                for c in range(vocals.shape[0])
            ])
        if instr.ndim == 1:
            instr = instr[np.newaxis, :]
        n = min(instr.shape[-1], vocals.shape[-1])
        return i_gain * instr[:, :n] + v_gain * vocals[:, :n]

    def _sdedit_segment(self, pipe, src, strength: float, prompt: str,
                        seed: Optional[int]):
        """SDEdit one audio segment (torch [2,N] @ 48k) -> (numpy [2,M], n_steps).

        No audio_tokenizer in this checkpoint means ACE-Step's `cover` task is
        unavailable, so we img2img: VAE-encode the source, noise it to sigma≈
        `strength`, and denoise the truncated flow-matching schedule with the
        source as timbre reference. `variation_amount` = the noise level.
        """
        import torch

        device = self._device
        # 1. Encode the source segment into the acoustic latent space.
        src_b = src.unsqueeze(0).to(device=device, dtype=pipe.vae.dtype)  # [1,2,N]
        with torch.no_grad():
            src_latents = pipe.vae.encode(src_b).latent_dist.sample().transpose(1, 2)
        model_dtype = pipe.transformer.dtype
        src_latents = src_latents.to(dtype=model_dtype)  # [1, T, D]
        latent_len = src_latents.shape[1]

        # 2. Build an SDEdit schedule of `target_steps` sigmas spanning [strength, 0],
        #    then apply the flow-matching shift. Constructing it directly (rather than
        #    slicing the full schedule) keeps the step count fixed at any strength —
        #    the shift otherwise clusters sigmas at the high end and starves low
        #    variation of steps.
        target_steps = max(2, int(os.environ.get("ACESTEP_STEPS", "8")))
        shift = 3.0
        base = torch.linspace(strength, 0.0, target_steps + 1, device=device,
                              dtype=torch.float32)[:-1]
        sched = shift * base / (1 + (shift - 1) * base)
        sigma0 = float(sched[0])

        # 3. Noise the source latents to sigma0: x = (1-σ)·x0 + σ·ε (flow matching).
        gen = torch.Generator(device=device).manual_seed(int(seed)) if seed is not None else None
        noise = torch.randn(src_latents.shape, generator=gen, device=device, dtype=model_dtype)
        x_init = (1.0 - sigma0) * src_latents + sigma0 * noise

        # 4. Denoise via the pipeline (source timbre + our prompt).
        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                lyrics="",
                audio_duration=latent_len / pipe.latents_per_second,
                task_type="text2music",
                reference_audio=src,
                latents=x_init,
                timesteps=sched.tolist(),
                num_inference_steps=sched.numel(),
                generator=gen,
            )
        samples = out.audios[0].detach().to("cpu").float().numpy()
        del src_latents, x_init, noise, out
        torch.cuda.empty_cache()
        return samples, sched.numel()

    @staticmethod
    def _crossfade_concat(pieces: list, overlap: int) -> np.ndarray:
        """Concatenate [2,N] segments with an equal-power crossfade over `overlap`."""
        if not pieces:
            return np.zeros((2, 0), dtype=np.float32)
        out = pieces[0]
        for nxt in pieces[1:]:
            ov = min(overlap, out.shape[-1], nxt.shape[-1])
            if ov <= 0:
                out = np.concatenate([out, nxt], axis=-1)
                continue
            t = np.linspace(0.0, 1.0, ov, dtype=np.float32)
            fade_out, fade_in = np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
            blended = out[:, -ov:] * fade_out + nxt[:, :ov] * fade_in
            out = np.concatenate([out[:, :-ov], blended, nxt[:, ov:]], axis=-1)
        return out

    @staticmethod
    def _to_model_tensor(clip: Clip, target_sr: int):
        """Resample to target_sr and force stereo -> torch tensor [2, N]."""
        import torch

        data = clip.samples.astype(np.float32)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        # Resample if needed.
        if clip.sample_rate and clip.sample_rate != target_sr:
            import librosa

            data = np.stack([
                librosa.resample(data[i], orig_sr=clip.sample_rate, target_sr=target_sr)
                for i in range(data.shape[0])
            ])
        # Force stereo (the pipeline expects 2 channels).
        if data.shape[0] == 1:
            data = np.repeat(data, 2, axis=0)
        elif data.shape[0] > 2:
            data = data[:2]
        return torch.from_numpy(np.ascontiguousarray(data)).float()
