"""Read clips from the R2 Lance frame datasets (one row per frame: clip_id, frame_index, jpeg, lmk, size)."""
from dataclasses import dataclass

import cv2
import numpy as np
import pyarrow.compute as pc

from emotion_lab import r2

DATASETS = {'dataset_clip': 's3://fullbodydataset/dataset_clip/frames.lance',
            'talkvid': 's3://fullbodydataset/talkvid/lance/frames.lance'}
FPS = 25.0  # not stored in the datasets; assumed


def open_dataset(name):
    import lance
    return lance.dataset(DATASETS[name], storage_options=r2.lance_storage_options())


def landmarks(lmk_bytes):
    """68 (x, y) points, stored as little-endian int32 pixel coords."""
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
    rows: list  # dicts: frame_index, jpeg, lmk, frame_width, frame_height (sorted by frame_index)

    def frames(self):
        for r in self.rows:
            yield cv2.imdecode(np.frombuffer(r['jpeg'], np.uint8), cv2.IMREAD_COLOR)

    def boxes(self):
        """Per-frame face box or None when the landmarks are missing/invalid."""
        out = []
        for r in self.rows:
            pts = landmarks(r['lmk'])
            w, h = r['frame_width'], r['frame_height']
            out.append(face_box(pts, w, h) if valid_landmarks(pts, w, h) else None)
        return out


def load_clip(dataset, clip_id, fragment):
    frag = open_dataset(dataset).get_fragments()[fragment]
    rows = frag.to_table(columns=['frame_index', 'jpeg', 'lmk', 'frame_width', 'frame_height'],
                         filter=pc.field('clip_id') == clip_id).sort_by('frame_index').to_pylist()
    if not rows:
        raise LookupError(f'{dataset}/{clip_id} not found in fragment {fragment}')
    return Clip(dataset, clip_id, fragment, rows)
