// Mirrors the backend Pydantic models / controls manifest.

export type ControlKind = "bipolar" | "unipolar" | "enum" | "multi";

export interface ControlSpec {
  key: string;
  label: string;
  kind: ControlKind;
  group: string;
  left: string;
  right: string;
  default: number;
  options: string[];
  help: string;
}

export interface ControlsManifest {
  groups: { key: string; label: string }[];
  controls: ControlSpec[];
  identityElements: string[];
}

export type ControlValue = number | string | string[];

export interface Steering {
  controls: Record<string, ControlValue>;
  locks: string[];
}

export interface AudioFeatures {
  duration_sec: number;
  sample_rate: number;
  tempo_bpm: number | null;
  key: string | null;
  rms_energy: number | null;
  spectral_centroid: number | null;
  embedding: number[];
  analyzed: boolean;
  note: string;
}

export interface Variant {
  id: string;
  song_id: string;
  parent_id: string | null;
  label: string;
  is_original: boolean;
  created_at: number;
  steering: Steering;
  features: AudioFeatures;
  similarity: number | null;
  audio_path: string | null;
  generator: string;
  rating: number | null;
}

export interface Song {
  id: string;
  title: string;
  original_filename: string;
  created_at: number;
  root_variant_id: string | null;
}

export interface TreeNode {
  variant: Variant;
  children: TreeNode[];
}

export interface Backend {
  name: string;
  description: string;
  available: boolean;
}

export interface Health {
  status: string;
  version: string;
  audio_backend: boolean;
  backends: Backend[];
}

export interface Job {
  id: string;
  kind: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  message: string;
  song_id: string | null;
  result: Variant | null;
  error: string | null;
  created_at: number;
  updated_at: number;
}

export interface LivingSegment {
  id: string;
  song_id: string;
  audio_url: string;
  duration: number;
  start_index: number;
  next_index: number;
  next_pos: number;
  advance: number;
  windows: number;
  note: string;
}

export interface PresetParams {
  improvisation: number;
  duration_sec: number;
  controls: Record<string, ControlValue>;
}

export interface Preset {
  id: string;
  name: string;
  builtin: boolean;
  description: string;
  params: PresetParams;
}

export interface LivingParams {
  duration_sec: number;
  improvisation: number;
  start_index: number;
  start_pos: number;
  seed: number;
  steering: Steering;
}

export interface Preferences {
  song_id: string;
  rated: number;
  loved: number;
  disliked: number;
  preferred_variation: number | null;
  preferred_similarity: number | null;
  avoided_variation: number | null;
  suggested_variation: number;
}
