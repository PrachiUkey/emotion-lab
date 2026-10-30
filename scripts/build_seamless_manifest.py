"""Pick an evaluation set of Seamless Interaction clips -> CSV manifest (dataset, clip_id, fragment, speaking).

    python scripts/build_seamless_manifest.py --n 60 --out manifests/seamless_v0.csv

Rules: complete clips of 3-10 s, at most one clip per recording (file_id), face landmarks on >= 80% of frames,
precomputed emotion scores on >= 50%, half "talking" (speaking on >= 50% of frames) and half "listening" (<= 10%).
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotion_lab.data import get_fragment, landmarks, open_dataset, valid_landmarks  # noqa: E402


def recording(clip_id):
    return clip_id.rsplit('_', 1)[0]  # si_<file_id>_<segment>


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--out', default='manifests/seamless_v0.csv')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min_frames', type=int, default=75)
    ap.add_argument('--max_frames', type=int, default=250)
    args = ap.parse_args()

    random.seed(args.seed)
    # fragment ids are 0..n-1 here; listing 412k fragments is slow, so count them and fetch by id
    n_frags = open_dataset('seamless').stats.dataset_stats()['num_fragments']
    want = {'talking': args.n // 2, 'listening': args.n - args.n // 2}
    picked, used, scanned = [], set(), 0
    for fi in random.sample(range(n_frags), n_frags):
        if not any(want.values()):
            break
        scanned += 1
        t = get_fragment('seamless', fi).to_table(columns=['clip_id', 'frame_index', 'lmk_seamless', 'frame_width', 'frame_height',
                                        'emotion_scores', 'speaking']).to_pylist()
        by_clip = {}
        for r in t:
            by_clip.setdefault(r['clip_id'], []).append(r)
        for cid, rows in random.sample(sorted(by_clip.items()), len(by_clip)):
            n = len(rows)
            idx = sorted(r['frame_index'] for r in rows)
            if recording(cid) in used or not (args.min_frames <= n <= args.max_frames) or idx != list(range(n)):
                continue
            face = np.mean([valid_landmarks(landmarks(r['lmk_seamless']), r['frame_width'], r['frame_height'])
                            for r in rows if r['lmk_seamless']] or [0]) * (sum(bool(r['lmk_seamless']) for r in rows) / n)
            emo = np.mean([bool(r['emotion_scores']) and any(r['emotion_scores']) for r in rows])
            talk = np.mean([bool(r['speaking']) for r in rows])
            kind = 'talking' if talk >= 0.5 else 'listening' if talk <= 0.1 else None
            if face < 0.8 or emo < 0.5 or kind is None or want[kind] == 0:
                continue
            picked.append({'dataset': 'seamless', 'clip_id': cid, 'fragment': fi, 'speaking': kind,
                           'n_frames': n, 'face_frac': round(float(face), 3), 'emotion_frac': round(float(emo), 3)})
            used.add(recording(cid))
            want[kind] -= 1
            break  # one clip per fragment keeps the set diverse
        print(f'scanned {scanned} fragments, picked {len(picked)}/{args.n}', flush=True)

    if any(want.values()):
        sys.exit(f'!!! only found {len(picked)} clips matching the rules (missing {want})')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(picked[0]))
        w.writeheader()
        w.writerows(picked)
    print(f'wrote {len(picked)} clips -> {args.out}')


if __name__ == '__main__':
    main()
