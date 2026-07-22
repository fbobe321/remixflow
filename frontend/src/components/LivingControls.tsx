import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { ControlsManifest, ControlValue, Preset, Song, Steering } from "../types";
import { SteeringControl } from "./SteeringControls";

interface Props {
  manifest: ControlsManifest;
  steering: Steering;
  improvisation: number;
  duration: number;
  songs: Song[];
  playlist: string[];
  perSongSec: number;
  onSteering: (s: Steering) => void;
  onImprovisation: (v: number) => void;
  onDuration: (v: number) => void;
  onPerSongSec: (v: number) => void;
  onTogglePlaylist: (songId: string) => void;
}

// A focused subset of the steering catalog that shapes the Living "feel".
const STYLE_KEYS = ["energy", "brightness", "warmth", "rock", "jazz", "electronic"];

export function LivingControls({
  manifest,
  steering,
  improvisation,
  duration,
  songs,
  playlist,
  perSongSec,
  onSteering,
  onImprovisation,
  onDuration,
  onPerSongSec,
  onTogglePlaylist,
}: Props) {
  const styleControls = manifest.controls.filter((c) => STYLE_KEYS.includes(c.key));
  const setControl = (key: string, value: ControlValue) =>
    onSteering({ ...steering, controls: { ...steering.controls, [key]: value } });

  const [presets, setPresets] = useState<Preset[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [busy, setBusy] = useState(false);

  const loadPresets = useCallback(() => {
    api.listPresets().then(setPresets).catch(() => {});
  }, []);
  useEffect(() => loadPresets(), [loadPresets]);

  const applyPreset = (p: Preset) => {
    setActiveId(p.id);
    onImprovisation(p.params.improvisation);
    onDuration(p.params.duration_sec);
    // Reset the style subset, then apply the preset's controls.
    const cleared: Record<string, ControlValue> = { ...steering.controls };
    for (const k of STYLE_KEYS) {
      const spec = manifest.controls.find((c) => c.key === k);
      if (spec) cleared[k] = spec.kind === "multi" ? [] : spec.default;
    }
    onSteering({ ...steering, controls: { ...cleared, ...p.params.controls } });
  };

  const savePreset = async () => {
    const name = window.prompt("Name this preset:");
    if (!name) return;
    setBusy(true);
    try {
      const controls: Record<string, ControlValue> = {};
      for (const k of STYLE_KEYS) if (k in steering.controls) controls[k] = steering.controls[k];
      const p = await api.createPreset(name, {
        improvisation,
        duration_sec: duration,
        controls,
      });
      await api.listPresets().then(setPresets);
      setActiveId(p.id);
    } catch {
      /* ignore */
    } finally {
      setBusy(false);
    }
  };

  const deletePreset = async (id: string) => {
    await api.deletePreset(id).catch(() => {});
    if (activeId === id) setActiveId("");
    loadPresets();
  };

  const builtins = presets.filter((p) => p.builtin);
  const mine = presets.filter((p) => !p.builtin);

  return (
    <div className="steering-panel">
      <div className="panel-header">
        <h2>Living controls</h2>
        <button className="ghost" onClick={savePreset} disabled={busy}>
          {busy ? "Saving…" : "Save preset"}
        </button>
      </div>

      <section className="control-group">
        <h3>Listening mode</h3>
        <div className="chip-row">
          {builtins.map((p) => (
            <button
              key={p.id}
              className={`chip ${activeId === p.id ? "active" : ""}`}
              title={p.description}
              onClick={() => applyPreset(p)}
            >
              {p.name}
            </button>
          ))}
        </div>
        {mine.length > 0 && (
          <>
            <p className="subtle" style={{ margin: "10px 0 6px" }}>Your presets</p>
            <div className="chip-row">
              {mine.map((p) => (
                <span key={p.id} className={`chip preset-user ${activeId === p.id ? "active" : ""}`}>
                  <button className="preset-apply" onClick={() => applyPreset(p)}>{p.name}</button>
                  <button className="preset-del" title="Delete" onClick={() => deletePreset(p.id)}>✕</button>
                </span>
              ))}
            </div>
          </>
        )}
      </section>

      <section className="control-group">
        <h3>Playlist {playlist.length > 1 ? `· ${playlist.length} songs` : ""}</h3>
        <p className="subtle" style={{ marginTop: 0 }}>
          Pick songs to weave into one endless Living set (transitions between them).
        </p>
        <div className="playlist-rows">
          {songs.map((s) => {
            const on = playlist.includes(s.id);
            const order = on ? playlist.indexOf(s.id) + 1 : null;
            return (
              <label key={s.id} className={`playlist-row ${on ? "on" : ""}`}>
                <input type="checkbox" checked={on} onChange={() => onTogglePlaylist(s.id)} />
                <span className="pl-order">{order ?? "·"}</span>
                <span className="pl-title">{s.title}</span>
              </label>
            );
          })}
        </div>
        {playlist.length > 1 && (
          <div className="control slider" style={{ marginTop: 12 }}>
            <div className="control-head">
              <span className="control-label">Time per song</span>
              <span className="control-readout">{Math.round(perSongSec)}s</span>
            </div>
            <input
              type="range"
              min={45}
              max={240}
              step={15}
              value={perSongSec}
              style={{ ["--fill" as string]: `${((perSongSec - 45) / 195) * 100}%` }}
              onChange={(e) => onPerSongSec(Number(e.target.value))}
            />
            <div className="control-ends">
              <span>Quick rotation</span>
              <span>Linger longer</span>
            </div>
          </div>
        )}
      </section>

      <section className="control-group">
        <h3>Performance</h3>
        <div className="control-grid">
          <div className="control slider">
            <div className="control-head">
              <span className="control-label">Improvisation</span>
              <span className="control-readout">{Math.round(improvisation * 100)}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={improvisation}
              style={{ ["--fill" as string]: `${improvisation * 100}%` }}
              onChange={(e) => onImprovisation(Number(e.target.value))}
            />
            <div className="control-ends">
              <span>Faithful</span>
              <span>Adventurous</span>
            </div>
          </div>

          <div className="control slider">
            <div className="control-head">
              <span className="control-label">Segment length</span>
              <span className="control-readout">{Math.round(duration)}s</span>
            </div>
            <input
              type="range"
              min={20}
              max={60}
              step={5}
              value={duration}
              style={{ ["--fill" as string]: `${((duration - 20) / 40) * 100}%` }}
              onChange={(e) => onDuration(Number(e.target.value))}
            />
            <div className="control-ends">
              <span>Snappier updates</span>
              <span>Longer stretches</span>
            </div>
          </div>
        </div>
      </section>

      <section className="control-group">
        <h3>Style drift</h3>
        <div className="control-grid">
          {styleControls.map((spec) => (
            <SteeringControl
              key={spec.key}
              spec={spec}
              value={steering.controls[spec.key]}
              onChange={setControl}
            />
          ))}
        </div>
      </section>

      <p className="subtle">
        Changes apply to the next generated stretch — the performance keeps
        flowing while you steer it.
      </p>
    </div>
  );
}
