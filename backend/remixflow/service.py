"""Application service layer: orchestrates import, generation, morphing, and
preference learning on top of the store, analysis, and generator backends."""

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np

from .audio import analysis
from .audio.io import Clip, AudioUnavailable, load, save
from .generation import get_generator
from .models import (
    AudioFeatures,
    GenerateRequest,
    LivingRequest,
    MorphRequest,
    Song,
    Steering,
    Variant,
)
from .store import Store


class ServiceError(RuntimeError):
    pass


class RemixService:
    def __init__(self, store: Store) -> None:
        self.store = store

    # --- import (PRD §1) ---------------------------------------------------

    def import_song(self, tmp_path: str, title: str, filename: str) -> tuple[Song, Variant]:
        try:
            clip = load(tmp_path)
        except AudioUnavailable as exc:
            raise ServiceError(str(exc)) from exc

        song = Song(title=title or filename or "Untitled", original_filename=filename)
        features = analysis.analyze(clip)
        root = Variant(
            song_id=song.id,
            parent_id=None,
            label="Original",
            is_original=True,
            features=features,
            similarity=1.0,
            generator="source",
        )
        # Persist the original audio (normalized to WAV) as the tree root.
        out_path = self.store.audio_path_for(root.id)
        save(clip, str(out_path))
        root.audio_path = str(out_path)

        self.store.add_song(song, root)
        return song, root

    # --- generation (PRD §3, §5, §6) --------------------------------------

    def _load_variant_clip(self, variant: Variant) -> Clip:
        if not variant.audio_path:
            raise ServiceError(f"Variant {variant.id} has no audio.")
        return load(variant.audio_path)

    def _root_variant(self, song_id: str) -> Optional[Variant]:
        song = self.store.get_song(song_id)
        if not song or not song.root_variant_id:
            return None
        return self.store.get_variant(song.root_variant_id)

    def generate(self, req: GenerateRequest, backend: str | None = None,
                 seed: int | None = None) -> Variant:
        parent = self.store.get_variant(req.parent_id)
        if parent is None:
            raise ServiceError(f"Unknown parent variant {req.parent_id!r}.")

        parent_clip = self._load_variant_clip(parent)
        root = self._root_variant(parent.song_id)
        root_clip = self._load_variant_clip(root) if root else None

        gen = get_generator(backend)
        try:
            result = gen.generate(parent_clip, req.steering, original=root_clip, seed=seed)
        except Exception as exc:  # a backend failure shouldn't 500 opaquely
            raise ServiceError(f"Generation failed ({gen.name}): {exc}") from exc

        variant = Variant(
            song_id=parent.song_id,
            parent_id=parent.id,
            label=req.label or self._auto_label(req.steering),
            steering=req.steering.normalized(),
            generator=f"{result.generator}: {result.note}",
        )
        out_path = self.store.audio_path_for(variant.id)
        save(result.clip, str(out_path))
        variant.audio_path = str(out_path)

        # Evaluate identity preservation against the ORIGINAL (re-anchoring).
        variant.features = analysis.analyze(result.clip)
        if root_clip is not None:
            variant.similarity = analysis.similarity(
                variant.features.embedding, root.features.embedding
            )
        self.store.add_variant(variant)
        return variant

    def morph(self, req: MorphRequest) -> Variant:
        """Blend two variants into an intermediate (PRD Advanced: Morph)."""
        a = self.store.get_variant(req.variant_a)
        b = self.store.get_variant(req.variant_b)
        if a is None or b is None:
            raise ServiceError("Both variants must exist to morph.")
        if a.song_id != b.song_id:
            raise ServiceError("Can only morph variants of the same song.")

        ca, cb = self._load_variant_clip(a), self._load_variant_clip(b)
        blend = max(0.0, min(1.0, req.blend))
        n = min(ca.samples.shape[-1], cb.samples.shape[-1])
        # Align channel counts.
        sa = ca.samples[..., :n]
        sb = cb.samples[..., :n]
        if sa.ndim != sb.ndim:
            sa = ca.to_mono()[:n][np.newaxis, :]
            sb = cb.to_mono()[:n][np.newaxis, :]
        mixed = (1.0 - blend) * sa + blend * sb
        clip = Clip(samples=mixed.astype(np.float32), sample_rate=ca.sample_rate)

        variant = Variant(
            song_id=a.song_id,
            parent_id=a.id,
            label=f"Morph {int((1 - blend) * 100)}/{int(blend * 100)}",
            steering=req.steering.normalized() if req.steering else Steering(),
            generator=f"morph({a.id[:8]},{b.id[:8]})",
        )
        out_path = self.store.audio_path_for(variant.id)
        save(clip, str(out_path))
        variant.audio_path = str(out_path)
        variant.features = analysis.analyze(clip)
        root = self._root_variant(a.song_id)
        if root:
            variant.similarity = analysis.similarity(
                variant.features.embedding, root.features.embedding
            )
        self.store.add_variant(variant)
        return variant

    # --- Living Mode (PRD Phase 2) ----------------------------------------

    def living_segment(self, req: "LivingRequest", *, backend: str | None = None,
                       progress=None) -> dict:
        """Render one segment of a continuously-evolving Living performance and
        persist it. ``start_index``/``next_index`` let the client chain segments
        seamlessly for endless 'Living Repeat' playback."""
        from .living import LivingConfig, LivingEngine, TensionCurve

        root = self._root_variant(req.song_id)
        if root is None:
            raise ServiceError(f"Unknown song {req.song_id!r}.")
        clip = self._load_variant_clip(root)

        # Improvisation → tension range. Calibrated so the default (~0.35) gives
        # audible "breathing" (like the tuned 0.15–0.42 demo) and the top end is
        # adventurous, while the low end stays near-identical.
        imp = max(0.0, min(1.0, req.improvisation))
        tension = TensionCurve(lo=0.08 + 0.12 * imp, hi=0.20 + 0.35 * imp,
                               period=5, jitter=0.04)
        cfg = LivingConfig(
            duration_sec=max(8.0, req.duration_sec),
            steering=req.steering.normalized(),
            tension=tension,
            base_seed=req.seed,
            preserve_vocals=True,
        )
        engine = LivingEngine(get_generator(backend))
        try:
            result = engine.render(clip, cfg, start_index=req.start_index,
                                   start_pos_sec=req.start_pos, progress=progress)
        except Exception as exc:
            raise ServiceError(f"Living render failed: {exc}") from exc

        seg_id = f"live_{uuid.uuid4().hex[:12]}"
        out_path = self.store.audio_dir / f"{seg_id}.wav"
        save(result.clip, str(out_path))
        return {
            "id": seg_id,
            "song_id": req.song_id,
            "audio_url": f"/api/living/audio/{seg_id}",
            "duration": round(result.clip.duration, 2),
            "start_index": req.start_index,
            "next_index": result.next_index,
            "next_pos": result.next_pos_sec,
            "advance": result.advance_sec,   # timeline advance (< duration by the crossfade tail)
            "windows": len(result.windows),
            "note": result.note,
        }

    # --- preference learning (PRD §8) -------------------------------------

    def rate(self, variant_id: str, rating: int) -> Variant:
        variant = self.store.get_variant(variant_id)
        if variant is None:
            raise ServiceError(f"Unknown variant {variant_id!r}.")
        variant.rating = max(-1, min(1, int(rating)))
        self.store.update_variant(variant)
        return variant

    def preference_profile(self, song_id: str) -> dict[str, object]:
        """Learn the user's preferred familiarity/novelty balance from ratings.

        A transparent baseline (PRD §8, §"User Trust"): average the steering of
        loved variants and contrast with disliked ones. A learned preference
        model replaces this while keeping the same shape.
        """
        variants = self.store.variants_for_song(song_id)
        loved = [v for v in variants if v.rating == 1]
        disliked = [v for v in variants if v.rating == -1]

        def avg_variation(vs: list[Variant]) -> Optional[float]:
            vals = [float(v.steering.controls.get("variation_amount", 0.0)) for v in vs]
            return round(sum(vals) / len(vals), 3) if vals else None

        def avg_similarity(vs: list[Variant]) -> Optional[float]:
            vals = [v.similarity for v in vs if v.similarity is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        return {
            "song_id": song_id,
            "rated": len(loved) + len(disliked),
            "loved": len(loved),
            "disliked": len(disliked),
            "preferred_variation": avg_variation(loved),
            "preferred_similarity": avg_similarity(loved),
            "avoided_variation": avg_variation(disliked),
            "suggested_variation": avg_variation(loved)
            or round(sum(float(v.steering.controls.get("variation_amount", 0.2))
                        for v in variants) / len(variants), 3) if variants else 0.2,
        }

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _auto_label(steering: Steering) -> str:
        amt = int(float(steering.controls.get("variation_amount", 0.2)) * 100)
        return f"Variation {amt}%"
