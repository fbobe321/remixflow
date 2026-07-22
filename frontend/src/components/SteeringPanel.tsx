import { useMemo } from "react";
import type { ControlsManifest, ControlValue, Steering, Variant } from "../types";
import { SteeringControl } from "./SteeringControls";

interface Props {
  manifest: ControlsManifest;
  steering: Steering;
  parent: Variant | null;
  busy: boolean;
  onChange: (steering: Steering) => void;
  onGenerate: () => void;
  onReset: () => void;
}

/** The "music steering" surface: grouped sliders, identity locks, Generate. */
export function SteeringPanel({
  manifest,
  steering,
  parent,
  busy,
  onChange,
  onGenerate,
  onReset,
}: Props) {
  const byGroup = useMemo(() => {
    const map: Record<string, typeof manifest.controls> = {};
    for (const c of manifest.controls) (map[c.group] ??= []).push(c);
    return map;
  }, [manifest]);

  const setControl = (key: string, value: ControlValue) =>
    onChange({ ...steering, controls: { ...steering.controls, [key]: value } });

  const toggleLock = (el: string) => {
    const locks = steering.locks.includes(el)
      ? steering.locks.filter((l) => l !== el)
      : [...steering.locks, el];
    onChange({ ...steering, locks });
  };

  return (
    <div className="steering-panel">
      <div className="panel-header">
        <div>
          <h2>Steering</h2>
          <p className="subtle">
            Evolving from <strong>{parent ? parent.label : "—"}</strong>
          </p>
        </div>
        <button className="ghost" onClick={onReset}>
          Reset
        </button>
      </div>

      {manifest.groups.map((g) =>
        byGroup[g.key] ? (
          <section key={g.key} className="control-group">
            <h3>{g.label}</h3>
            <div className="control-grid">
              {byGroup[g.key].map((spec) => (
                <SteeringControl
                  key={spec.key}
                  spec={spec}
                  value={steering.controls[spec.key]}
                  onChange={setControl}
                />
              ))}
            </div>
          </section>
        ) : null
      )}

      <section className="control-group">
        <h3>Keep Fixed (Identity)</h3>
        <p className="subtle">
          Locked elements are preserved; everything else may evolve. Locking{" "}
          <strong>lyrics</strong> or <strong>vocal phrasing</strong> keeps your
          original vocal recording (the instrumental is varied around it).
        </p>
        <div className="chip-row">
          {manifest.identityElements.map((el) => (
            <button
              key={el}
              className={`chip lock ${steering.locks.includes(el) ? "active" : ""}`}
              onClick={() => toggleLock(el)}
            >
              {steering.locks.includes(el) ? "🔒 " : ""}
              {el.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      </section>

      <button className="generate" disabled={busy || !parent} onClick={onGenerate}>
        {busy ? "Generating…" : "⚡ Generate Variation"}
      </button>
    </div>
  );
}
