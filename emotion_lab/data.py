"""Read clips from the R2 Lance frame datasets (one row per frame: clip_id, frame_index, jpeg, landmarks, size)."""
import json
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
import pyarrow.compute as pc

from emotion_lab import r2

FPS = 25.0  # fallback when a dataset does not record the frame rate

SEAMLESS = 's3://semless-dataset/seamless_lance'
FULLBODY = 's3://fullbodydataset'
# frames: one row per frame. timing: real fps + frame count per clip. storage: which clips have audio
# (wav_bytes > 0) and where (wav_key). audio_key: path pattern when storage has no key column.
DATASETS = {
    'dataset_clip': {'frames': f'{FULLBODY}/dataset_clip/frames.lance', 'timing': f'{FULLBODY}/dataset_clip/timing.lance',
                     'storage': f'{FULLBODY}/dataset_clip/storage.lance', 'audio_bucket': 'fullbodydataset'},
    'talkvid': {'frames': f'{FULLBODY}/talkvid/lance/frames.lance', 'timing': f'{FULLBODY}/talkvid/lance/timing.lance',
                'storage': f'{FULLBODY}/talkvid/lance/storage.lance', 'audio_bucket': 'fullbodydataset'},
    # Meta Seamless Interaction: full-body portrait frames, transcripts, precomputed per-frame emotion scores.
    # `lmk` is often empty there; `lmk_seamless` holds the points.
    'seamless': {'frames': f'{SEAMLESS}/frames_full.lance', 'extra_columns': ['lmk_seamless', 'emotion_scores', 'speaking'],
                 'timing': f'{SEAMLESS}/timing.lance', 'meta': f'{SEAMLESS}/seamless.lance',
                 'storage': f'{SEAMLESS}/storage.lance', 'audio_bucket': 'semless-dataset',
                 'audio_key': 'seamless_lance/audio/{clip_id}.wav'},
}
BASE_COLUMNS = ['frame_index', 'jpeg', 'lmk', 'frame_width', 'frame_height']


@lru_cache(maxsize=None)
def _open(uri):
    import lance
    return lance.dataset(uri, storage_options=r2.lance_storage_options())


def open_dataset(name):
    return _open(DATASETS[name]['frames'])


def landmarks(lmk_bytes):
    """68 (x, y) points, stored as little-endian int32 pixel coords in the full frame."""
    return np.frombuffer(lmk_bytes, dtype='<i4').reshape(68, 2)


def valid_landmarks(pts, w, h):
    return bool(pts.any() and (pts >= 0).all() and (pts[:, 0] < w).all() and (pts[:, 1] < h).all()
                and np.ptp(pts[:, 0]) > 30)


def face_box(pts, w, h):
    """Landmark bbox widened 10% and extended 30% upward to include the forehead, clipped to the frame."""
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    bw, bh = x1 - x0, y1 - y0
    return [int(max(0, x0 - 0.1 * bw)), int(max(0, y0 - 0.3 * bh)),
            int(min(w, x1 + 0.1 * bw)), int(min(h, y1 + 0.05 * bh))]


@dataclass
class Clip:
    dataset: str
    clip_id: str
    fragment: int
    rows: list  # dicts with BASE_COLUMNS (+ dataset extras), sorted by frame_index
    fps: float = FPS
    fps_source: str = 'assumed'
    has_audio: bool = False
    warnings: list = field(default_factory=list)
    transcript: list = field(default_factory=list)  # [{'word', 'start', 'end'}], seconds from clip start

    def frames(self):
        for r in self.rows:
            yield cv2.imdecode(np.frombuffer(r['jpeg'], np.uint8), cv2.IMREAD_COLOR)

    def points(self):
        """Per-frame 68 landmarks (full-frame pixels) or None when missing/invalid."""
        out = []
        for r in self.rows:
            w, h = r['frame_width'], r['frame_height']
            pts = landmarks(r['lmk'])
            if not valid_landmarks(pts, w, h) and r.get('lmk_seamless'):
                pts = landmarks(r['lmk_seamless'])
            out.append(pts if valid_landmarks(pts, w, h) else None)
        return out

    def boxes(self):
        """Per-frame face box or None."""
        return [None if p is None else face_box(p, r['frame_width'], r['frame_height'])
                for p, r in zip(self.points(), self.rows)]

    def audio(self):
        """The clip's WAV bytes, or None when the dataset says it has no audio."""
        key = audio_key(self.dataset, self.clip_id)
        if key is None:
            return None
        return r2.s3_client().get_object(Bucket=DATASETS[self.dataset]['audio_bucket'], Key=key)['Body'].read()


def audio_key(dataset, clip_id):
    """Object key of the clip's audio from the storage table (wav_bytes > 0), else None."""
    cfg = DATASETS[dataset]
    if 'storage' not in cfg:
        return None
    st = _open(cfg['storage'])
    cols = ['wav_bytes'] + (['wav_key'] if 'wav_key' in st.schema.names else [])
    rows = st.to_table(columns=cols, filter=pc.field('id') == clip_id).to_pylist()
    if not rows or not rows[0]['wav_bytes']:
        return None
    return rows[0].get('wav_key') or cfg['audio_key'].format(clip_id=clip_id)


def _timing(dataset, clip_id):
    uri = DATASETS[dataset].get('timing')
    if not uri:
        return None
    ds = _open(uri)
    cols = ['fps', 'num_frames'] + (['start_seconds'] if 'start_seconds' in ds.schema.names else [])
    rows = ds.to_table(columns=cols, filter=pc.field('id') == clip_id).to_pylist()
    return rows[0] if rows else None


_TRANSCRIPTS = {}  # (dataset, clip_id) -> transcript JSON, filled by prefetch_transcripts()


def prefetch_transcripts(dataset, clip_ids):
    """Fetch transcripts for many clips in ONE scan. The seamless meta table is spread over many small files,
    so even a single-id lookup scans it all (~5 min); per-clip lookups would be unusable."""
    uri = DATASETS[dataset].get('meta')
    if not uri or not clip_ids:
        return 0
    rows = _open(uri).to_table(columns=['id', 'transcript'], filter=pc.field('id').isin(list(clip_ids))).to_pylist()
    for r in rows:
        _TRANSCRIPTS[(dataset, r['id'])] = r['transcript']
    return len(rows)


def _transcript(dataset, clip_id, clip_start):
    """Word timings (clip-relative seconds) if prefetched, else [] — never a slow per-clip scan."""
    raw = _TRANSCRIPTS.get((dataset, clip_id))
    if not raw:
        return []
    words = []
    for seg in json.loads(raw):
        for w in seg.get('words', []):
            words.append({'word': w['word'], 'start': round(w['start'] - clip_start, 3), 'end': round(w['end'] - clip_start, 3)})
    return words


def get_fragment(dataset, fragment_id):
    """Manifests store Lance fragment *ids*. Fetch the one fragment directly: listing all fragments
    (get_fragments()) takes minutes on these datasets (412k fragments for seamless)."""
    frag = open_dataset(dataset).get_fragment(fragment_id)
    if frag is None:
        raise LookupError(f'{dataset}: no fragment with id {fragment_id}')
    return frag


def load_clip(dataset, clip_id, fragment):
    cfg = DATASETS[dataset]
    frag = get_fragment(dataset, fragment)
    # Find the clip's rows from the small clip_id column, then fetch only those rows: a filtered scan would
    # read every JPEG in the fragment (hundreds of MB on the full-resolution datasets).
    ids = frag.to_table(columns=['clip_id'])['clip_id']
    pos = np.flatnonzero(pc.equal(ids, clip_id).to_numpy(zero_copy_only=False)).tolist()
    rows = (frag.take(pos, columns=BASE_COLUMNS + cfg.get('extra_columns', [])).sort_by('frame_index').to_pylist()
            if pos else [])
    if not rows:
        raise LookupError(f'{dataset}/{clip_id} not found in fragment {fragment}')
    clip = Clip(dataset, clip_id, fragment, rows)
    t = _timing(dataset, clip_id)
    if t and t['fps']:
        if t['num_frames'] != len(rows):
            # seen as off-by-one on dataset_clip; keep the real fps, use the frames we actually have, record it
            clip.warnings.append(f'timing says {t["num_frames"]} frames, Lance has {len(rows)}')
            print(f'!!! {dataset}/{clip_id}: {clip.warnings[-1]}', flush=True)
        clip.fps, clip.fps_source = float(t['fps']), 'dataset'
        if t.get('start_seconds') is not None:
            clip.transcript = _transcript(dataset, clip_id, t['start_seconds'])
    clip.has_audio = audio_key(dataset, clip_id) is not None
    return clip
