import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import type { Variant } from "../types";

interface Props {
  variants: Variant[]; // ordered: original first, then variants
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRate: (id: string, rating: number) => void;
}

/** A/B comparison (PRD §7): instant switching between versions, preserving
 * playback position so differences are easy to hear. Plus ratings (PRD §8). */
export function ABPlayer({ variants, selectedId, onSelect, onRate }: Props) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const selected = useMemo(
    () => variants.find((v) => v.id === selectedId) ?? variants[0],
    [variants, selectedId]
  );

  // Swap source but keep the playhead — the essence of A/B auditioning.
  useEffect(() => {
    const el = audioRef.current;
    if (!el || !selected) return;
    const pos = el.currentTime;
    const wasPlaying = !el.paused;
    el.src = api.audioUrl(selected.id);
    el.load();
    el.currentTime = Number.isFinite(pos) ? pos : 0;
    if (wasPlaying) el.play().catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);

  if (!selected) return null;
  const feats = selected.features;

  return (
    <div className="ab-player">
      <div className="now-playing">
        <div>
          <div className="np-label">{selected.label}</div>
          <div className="subtle np-meta">
            {feats.tempo_bpm ? `${Math.round(feats.tempo_bpm)} BPM · ` : ""}
            {feats.key ? `Key ${feats.key} · ` : ""}
            {feats.duration_sec ? `${feats.duration_sec.toFixed(1)}s` : ""}
            {selected.generator ? ` · ${selected.generator}` : ""}
          </div>
        </div>
        {!selected.is_original && (
          <div className="rate-buttons">
            <button
              className={selected.rating === 1 ? "active" : ""}
              onClick={() => onRate(selected.id, 1)}
              title="Love it"
            >
              👍
            </button>
            <button
              className={selected.rating === 0 ? "active" : ""}
              onClick={() => onRate(selected.id, 0)}
              title="Neutral"
            >
              😐
            </button>
            <button
              className={selected.rating === -1 ? "active" : ""}
              onClick={() => onRate(selected.id, -1)}
              title="Don't like it"
            >
              👎
            </button>
          </div>
        )}
      </div>

      <audio
        ref={audioRef}
        controls
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        style={{ width: "100%" }}
      />

      <div className="ab-switcher">
        <span className="subtle">A/B:</span>
        {variants.map((v) => (
          <button
            key={v.id}
            className={`ab-chip ${v.id === selected.id ? "active" : ""}`}
            onClick={() => onSelect(v.id)}
          >
            {v.is_original ? "Original" : v.label}
          </button>
        ))}
      </div>
      <p className="subtle playing-hint">
        {playing ? "▶ Switch versions while playing to compare." : "Press play, then A/B switch."}
      </p>
    </div>
  );
}
