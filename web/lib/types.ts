export type ClipSummary = { label: string; conf: number };

export type IndexEntry = {
  dataset: string;
  clip_id: string;
  fragment: number;
  n_frames: number;
  face_frac: number;
  models: Record<string, ClipSummary>;
};

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
