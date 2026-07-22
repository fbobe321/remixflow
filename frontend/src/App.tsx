import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { ABPlayer } from "./components/ABPlayer";
import { EvolutionTree } from "./components/EvolutionTree";
import { Library } from "./components/Library";
import { LivingControls } from "./components/LivingControls";
import { LivingPlayer } from "./components/LivingPlayer";
import { SteeringPanel } from "./components/SteeringPanel";
import type {
  ControlsManifest,
  Health,
  Preferences,
  Song,
  Steering,
  TreeNode,
  Variant,
} from "./types";

function emptySteering(manifest: ControlsManifest): Steering {
  const controls: Steering["controls"] = {};
  for (const c of manifest.controls) {
    if (c.kind === "enum") controls[c.key] = c.options[0] ?? "";
    else if (c.kind === "multi") controls[c.key] = [];
    else controls[c.key] = c.default;
  }
  return { controls, locks: [] };
}

function flattenTree(node: TreeNode | null): Variant[] {
  if (!node) return [];
  return [node.variant, ...node.children.flatMap(flattenTree)];
}

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [manifest, setManifest] = useState<ControlsManifest | null>(null);
  const [songs, setSongs] = useState<Song[]>([]);
  const [songId, setSongId] = useState<string | null>(null);
  const [tree, setTree] = useState<TreeNode | null>(null);
  const [prefs, setPrefs] = useState<Preferences | null>(null);

  const [parentId, setParentId] = useState<string | null>(null); // branch source
  const [selectedId, setSelectedId] = useState<string | null>(null); // auditioned
  const [steering, setSteering] = useState<Steering | null>(null);

  const [busy, setBusy] = useState(false);
  const [importing, setImporting] = useState(false);
  const [infinite, setInfinite] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backend, setBackend] = useState<string | undefined>(undefined);
  const [progress, setProgress] = useState<{ pct: number; msg: string } | null>(null);
  const infiniteRef = useRef(false);

  // Living Mode state.
  const [mode, setMode] = useState<"steer" | "living">("steer");
  const [improvisation, setImprovisation] = useState(0.35);
  const [livingDuration, setLivingDuration] = useState(40);
  const [livingSeed, setLivingSeed] = useState(7);
  // Latest living params, read fresh each segment so steering mid-performance works.
  const livingParamsRef = useRef({ improvisation, livingDuration, livingSeed, steering });
  livingParamsRef.current = { improvisation, livingDuration, livingSeed, steering: steering ?? { controls: {}, locks: [] } };

  const variants = useMemo(() => flattenTree(tree), [tree]);
  const parent = useMemo(
    () => variants.find((v) => v.id === parentId) ?? null,
    [variants, parentId]
  );
  const selected = useMemo(
    () => variants.find((v) => v.id === selectedId) ?? null,
    [variants, selectedId]
  );

  // --- bootstrap -------------------------------------------------------
  useEffect(() => {
    Promise.all([api.health(), api.controls(), api.listSongs()])
      .then(([h, m, s]) => {
        setHealth(h);
        setManifest(m);
        setSteering(emptySteering(m));
        setSongs(s);
        if (s[0]) void selectSong(s[0].id);
      })
      .catch((e) => setError(String(e.message ?? e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshTree = useCallback(async (sid: string) => {
    const [t, p] = await Promise.all([api.tree(sid), api.preferences(sid)]);
    setTree(t);
    setPrefs(p);
    return t;
  }, []);

  const selectSong = useCallback(
    async (sid: string) => {
      setSongId(sid);
      setError(null);
      const t = await refreshTree(sid);
      setParentId(t.variant.id);
      setSelectedId(t.variant.id);
    },
    [refreshTree]
  );

  // --- actions ---------------------------------------------------------
  const handleImport = async (file: File, title: string) => {
    setImporting(true);
    setError(null);
    try {
      const { song } = await api.importSong(file, title);
      setSongs(await api.listSongs());
      await selectSong(song.id);
    } catch (e: any) {
      setError(String(e.message ?? e));
    } finally {
      setImporting(false);
    }
  };

  const handleDelete = async (sid: string) => {
    await api.deleteSong(sid);
    const list = await api.listSongs();
    setSongs(list);
    if (sid === songId) {
      setTree(null);
      setSongId(null);
      setParentId(null);
      setSelectedId(null);
      if (list[0]) void selectSong(list[0].id);
    }
  };

  const generateOnce = useCallback(
    async (fromId: string, s: Steering): Promise<Variant | null> => {
      try {
        const v = await api.generateAndWait(fromId, s, { backend }, (job) =>
          setProgress({ pct: Math.round(job.progress * 100), msg: job.message })
        );
        if (songId) await refreshTree(songId);
        setSelectedId(v.id);
        return v;
      } catch (e: any) {
        setError(String(e.message ?? e));
        return null;
      } finally {
        setProgress(null);
      }
    },
    [songId, refreshTree, backend]
  );

  const handleGenerate = async () => {
    if (!parentId || !steering) return;
    setBusy(true);
    await generateOnce(parentId, steering);
    setBusy(false);
  };

  // Infinite Evolution Mode (PRD §6): each variant branches from the last,
  // small changes chained forever until toggled off.
  useEffect(() => {
    infiniteRef.current = infinite;
    if (!infinite || !steering) return;
    let cancelled = false;
    (async () => {
      let from = selectedId ?? parentId;
      while (!cancelled && infiniteRef.current && from) {
        setBusy(true);
        const evolveStep: Steering = {
          ...steering,
          controls: { ...steering.controls, variation_amount: 0.12 }, // gentle drift
        };
        const v = await generateOnce(from, evolveStep);
        setBusy(false);
        if (!v) break;
        from = v.id;
        setParentId(v.id);
        await new Promise((r) => setTimeout(r, 1200));
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [infinite]);

  const handleRate = async (id: string, rating: number) => {
    await api.rate(id, rating);
    if (songId) await refreshTree(songId);
  };

  const resetSteering = () => manifest && setSteering(emptySteering(manifest));

  // --- render ----------------------------------------------------------
  if (error && !manifest) {
    return (
      <div className="fatal">
        <h1>RemixFlow</h1>
        <p>Could not reach the API: {error}</p>
        <p className="subtle">Is the backend running? `remixflow serve`</p>
      </div>
    );
  }
  if (!manifest || !steering) return <div className="loading">Loading RemixFlow…</div>;

  const similarity = selected?.similarity ?? (selected?.is_original ? 1 : null);
  const variationPct = Math.round((Number(steering.controls.variation_amount) || 0) * 100);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◈</span> RemixFlow
          <span className="tagline">AI music evolution</span>
        </div>
        {songId && (
          <div className="mode-switch" role="tablist">
            <button
              className={mode === "steer" ? "active" : ""}
              onClick={() => setMode("steer")}
            >
              Steer
            </button>
            <button
              className={mode === "living" ? "active" : ""}
              onClick={() => {
                if (mode !== "living") setLivingSeed(Math.floor(Math.random() * 100000));
                setMode("living");
              }}
            >
              ∞ Living
            </button>
          </div>
        )}
        <div className="health">
          {health && (
            <span className={`dot ${health.audio_backend ? "ok" : "warn"}`} title="Audio backend">
              {health.audio_backend ? "audio ✓" : "audio: limited"}
            </span>
          )}
        </div>
      </header>

      {error && <div className="error-banner" onClick={() => setError(null)}>{error} ✕</div>}

      <div className="layout">
        <aside className="col-left">
          <h2>Library</h2>
          <Library
            songs={songs}
            currentSongId={songId}
            busy={importing}
            onImport={handleImport}
            onSelect={selectSong}
            onDelete={handleDelete}
          />
        </aside>

        <main className="col-mid">
          {!songId ? (
            <div className="empty-hero">
              <h1>Keep everything you love. Just make it slightly different.</h1>
              <p className="subtle">Import a song to begin evolving it.</p>
            </div>
          ) : mode === "living" ? (
            <LivingPlayer
              key={songId}
              songId={songId}
              backend={backend}
              getParams={() => {
                const p = livingParamsRef.current;
                return {
                  duration_sec: p.livingDuration,
                  improvisation: p.improvisation,
                  seed: p.livingSeed,
                  steering: p.steering ?? { controls: {}, locks: [] },
                };
              }}
              onError={(m) => setError(m)}
            />
          ) : (
            <>
              <div className="meters">
                <Meter label="Similarity" value={similarity} accent="teal" />
                <Meter label="Variation" value={variationPct / 100} accent="violet" />
                {health && health.backends.length > 0 && (
                  <select
                    className="backend-select"
                    value={backend ?? ""}
                    onChange={(e) => setBackend(e.target.value || undefined)}
                    title="Generation backend"
                  >
                    <option value="">Auto</option>
                    {health.backends.map((b) => (
                      <option key={b.name} value={b.name} disabled={!b.available}>
                        {b.name}
                        {b.available ? "" : " (unavailable)"}
                      </option>
                    ))}
                  </select>
                )}
                <label className="infinite-toggle">
                  <input
                    type="checkbox"
                    checked={infinite}
                    onChange={(e) => setInfinite(e.target.checked)}
                  />
                  ∞ Evolution
                </label>
              </div>

              {progress && (
                <div className="progress-bar" title={progress.msg}>
                  <div className="progress-fill" style={{ width: `${progress.pct}%` }} />
                  <span className="progress-label">
                    {progress.msg || "Generating…"} {progress.pct}%
                  </span>
                </div>
              )}

              <ABPlayer
                variants={variants}
                selectedId={selectedId}
                onSelect={setSelectedId}
                onRate={handleRate}
              />

              <section className="tree-section">
                <div className="section-head">
                  <h2>Evolution Tree</h2>
                  {prefs && prefs.rated > 0 && (
                    <span className="subtle">
                      👍 {prefs.loved} · 👎 {prefs.disliked}
                      {prefs.preferred_variation != null &&
                        ` · you like ~${Math.round(prefs.preferred_variation * 100)}% variation`}
                    </span>
                  )}
                </div>
                <EvolutionTree
                  tree={tree}
                  selectedId={selectedId}
                  parentId={parentId}
                  onSelect={setSelectedId}
                  onBranchFrom={(id) => {
                    setParentId(id);
                    setSelectedId(id);
                  }}
                />
              </section>
            </>
          )}
        </main>

        <aside className="col-right">
          {songId && mode === "living" ? (
            <LivingControls
              manifest={manifest}
              steering={steering}
              improvisation={improvisation}
              duration={livingDuration}
              onSteering={setSteering}
              onImprovisation={setImprovisation}
              onDuration={setLivingDuration}
            />
          ) : songId ? (
            <SteeringPanel
              manifest={manifest}
              steering={steering}
              parent={parent}
              busy={busy}
              onChange={setSteering}
              onGenerate={handleGenerate}
              onReset={resetSteering}
            />
          ) : null}
        </aside>
      </div>
    </div>
  );
}

function Meter({
  label,
  value,
  accent,
}: {
  label: string;
  value: number | null;
  accent: string;
}) {
  const pct = value == null ? 0 : Math.round(value * 100);
  return (
    <div className={`meter ${accent}`}>
      <div className="meter-head">
        <span>{label}</span>
        <span>{value == null ? "—" : `${pct}%`}</span>
      </div>
      <div className="meter-track">
        <div className="meter-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
