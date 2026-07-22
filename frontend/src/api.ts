import type {
  ControlsManifest,
  Health,
  Job,
  LivingParams,
  LivingSegment,
  Preferences,
  Preset,
  PresetParams,
  Song,
  Steering,
  TreeNode,
  Variant,
} from "./types";

// Same-origin in production (FastAPI serves the built app); Vite proxies /api in dev.
const BASE = "";

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch(`${BASE}/api/health`).then(j<Health>),

  controls: () => fetch(`${BASE}/api/controls`).then(j<ControlsManifest>),

  listSongs: () => fetch(`${BASE}/api/songs`).then(j<Song[]>),

  importSong: (file: File, title: string) => {
    const form = new FormData();
    form.append("file", file);
    const url = `${BASE}/api/songs?title=${encodeURIComponent(title)}`;
    return fetch(url, { method: "POST", body: form }).then(
      j<{ song: Song; root: Variant }>
    );
  },

  deleteSong: (songId: string) =>
    fetch(`${BASE}/api/songs/${songId}`, { method: "DELETE" }).then(j),

  tree: (songId: string) => fetch(`${BASE}/api/songs/${songId}/tree`).then(j<TreeNode>),

  preferences: (songId: string) =>
    fetch(`${BASE}/api/songs/${songId}/preferences`).then(j<Preferences>),

  // Enqueue generation; returns a Job to poll.
  generate: (
    parentId: string,
    steering: Steering,
    opts: { label?: string; seed?: number; backend?: string } = {}
  ) => {
    const params = new URLSearchParams();
    if (opts.seed !== undefined) params.set("seed", String(opts.seed));
    if (opts.backend) params.set("backend", opts.backend);
    const qs = params.toString() ? `?${params}` : "";
    return fetch(`${BASE}/api/generate${qs}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parent_id: parentId, steering, label: opts.label ?? "" }),
    }).then(j<Job>);
  },

  job: (jobId: string) => fetch(`${BASE}/api/jobs/${jobId}`).then(j<Job>),

  // Submit a generation and poll to completion.
  generateAndWait: async (
    parentId: string,
    steering: Steering,
    opts: { label?: string; seed?: number; backend?: string } = {},
    onProgress?: (job: Job) => void
  ): Promise<Variant> => {
    let job = await api.generate(parentId, steering, opts);
    onProgress?.(job);
    while (job.status === "queued" || job.status === "running") {
      await new Promise((r) => setTimeout(r, 700));
      job = await api.job(job.id);
      onProgress?.(job);
    }
    if (job.status === "error" || !job.result) {
      throw new Error(job.error ?? "Generation failed");
    }
    return job.result;
  },

  morph: (a: string, b: string, blend: number) =>
    fetch(`${BASE}/api/morph`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant_a: a, variant_b: b, blend }),
    }).then(j<Variant>),

  rate: (variantId: string, rating: number) =>
    fetch(`${BASE}/api/variants/${variantId}/rate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating }),
    }).then(j<Variant>),

  audioUrl: (variantId: string) => `${BASE}/api/audio/${variantId}`,

  // --- Living Mode -----------------------------------------------------

  // Render one Living segment; returns a Variant-like segment once the job done.
  living: async (
    songId: string,
    params: LivingParams,
    opts: { backend?: string } = {},
    onProgress?: (job: Job) => void
  ): Promise<LivingSegment> => {
    const qs = opts.backend ? `?backend=${encodeURIComponent(opts.backend)}` : "";
    let job = await fetch(`${BASE}/api/living${qs}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ song_id: songId, ...params }),
    }).then(j<Job>);
    onProgress?.(job);
    while (job.status === "queued" || job.status === "running") {
      await new Promise((r) => setTimeout(r, 700));
      job = await api.job(job.id);
      onProgress?.(job);
    }
    if (job.status === "error" || !job.result) {
      throw new Error(job.error ?? "Living render failed");
    }
    return job.result as unknown as LivingSegment;
  },

  livingAudioUrl: (url: string) => `${BASE}${url}`,

  // --- presets ---------------------------------------------------------

  listPresets: () => fetch(`${BASE}/api/presets`).then(j<Preset[]>),

  createPreset: (name: string, params: PresetParams) =>
    fetch(`${BASE}/api/presets`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, params }),
    }).then(j<Preset>),

  deletePreset: (id: string) =>
    fetch(`${BASE}/api/presets/${id}`, { method: "DELETE" }).then(j),
};
