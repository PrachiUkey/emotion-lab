export type ClipSummary = { label: string; conf: number };

export type IndexEntry = {
  dataset: string;
  clip_id: string;
  fragment: number;
  n_frames: number;
  face_frac: number;
  fps?: number;
  has_audio?: boolean;
  models: Record<string, ClipSummary>;
};

export type ModelMeta = { classes: string[]; trained_on?: string; kind?: string; modality?: 'face' | 'audio' };
export type Word = { word: string; start: number; end: number };

export type PredsDoc = {
  dataset: string;
  clip_id: string;
  fps: number;
  n_frames: number;
  width: number;
  height: number;
  labels: string[];
  boxes: ([number, number, number, number] | null)[];
  frames: Record<string, (number[] | null)[]>;
  clip: Record<string, ClipSummary & { probs: number[] }>;
  face_frac: number;
  fps_source?: 'dataset' | 'assumed';
  transcript?: Word[];
  warnings?: string[];
  model_meta?: Record<string, ModelMeta>;
};

export type Gold = Record<string, { emotion: string; confidence?: string; annotator?: string; notes?: string }>;

export type LeaderRow = {
  model: string;
  n: number;
  macro_f1: number;
  macro_f1_ci95: [number, number];
  accuracy: number;
  recall: Record<string, number>;
  neutral_fp_rate: number | null;
  macro_f1_non_neutral?: number | null;
  n_non_neutral?: number;
  classes?: string[] | null;
  ece: number;
  flip_rate: number;
  confusion: number[][];
};

export type Leaderboard = {
  n_clips: number;
  labels: string[];
  gold_distribution: Record<string, number>;
  leaderboard: LeaderRow[];
  agreement: { cohen_kappa: Record<string, number>; fleiss_kappa: number | null };
  min_confidence: string;
};

export const EMOTION_COLORS: Record<string, string> = {
  neutral: '#8a94a6', happy: '#e8b422', sad: '#4a7bd0', surprised: '#d06fd0',
  fear: '#7a5bc4', disgusted: '#4fa36b', angry: '#e0533d', contempt: '#c98a4b',
};
