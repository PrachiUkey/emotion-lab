"""Run emotion models per frame on the clips of a manifest and write results to the store.

    python scripts/run_models.py --manifest gold/labels_v0.csv                 # all enabled models
    python scripts/run_models.py --manifest eval/manifest.csv --models vit_fer # add one model

Manifest: CSV with columns dataset, clip_id, fragment. Existing per-model results are kept; only missing
models are computed. Writes preds/, clips/ (mp4), thumbs/ and index.json.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotion_lab.data import load_clip  # noqa: E402
from emotion_lab.infer import encode_mp4, predict_clip, thumbnail  # noqa: E402
from emotion_lab.models.registry import load_models  # noqa: E402
from emotion_lab.store import get_json, open_store, put_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--models', default=None, help='comma-separated; default = all enabled in configs/models.yaml')
    ap.add_argument('--store', default=None, help='local:<dir> or r2:<bucket>/<prefix> (default $EMOTION_LAB_STORE)')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    store = open_store(args.store)
    models = load_models(args.models.split(',') if args.models else None)
    rows = list(csv.DictReader(open(args.manifest)))[:args.limit]
    index = {f"{e['dataset']}/{e['clip_id']}": e for e in (get_json(store, 'index.json')
                                                           if store.exists('index.json') else [])}
    print(f'store={store} models={[m.name for m in models]} clips={len(rows)}', flush=True)

    failures = []
    for n, row in enumerate(rows, 1):
        key = f"{row['dataset']}/{row['clip_id']}"
        t = time.time()
        try:
            clip = load_clip(row['dataset'], row['clip_id'], int(row['fragment']))
            pkey = f'preds/{key}.json'
            doc = predict_clip(clip, models, get_json(store, pkey) if store.exists(pkey) else None)
            put_json(store, pkey, doc)
            if not store.exists(f'clips/{key}.mp4'):
                store.put_bytes(f'clips/{key}.mp4', encode_mp4(clip), 'video/mp4')
            if not store.exists(f'thumbs/{key}.jpg'):
                store.put_bytes(f'thumbs/{key}.jpg', thumbnail(clip), 'image/jpeg')
            index[key] = {'dataset': row['dataset'], 'clip_id': row['clip_id'], 'fragment': int(row['fragment']),
                          'n_frames': doc['n_frames'], 'face_frac': doc['face_frac'],
                          'models': {k: {'label': v['label'], 'conf': v['conf']} for k, v in doc['clip'].items()}}
            summary = ' '.join(f"{k}={v['label']}:{v['conf']:.2f}" for k, v in doc['clip'].items())
            print(f'[{n}/{len(rows)}] {key} {time.time() - t:.1f}s {summary}', flush=True)
        except Exception as e:  # keep going, but report every failure at the end
            failures.append((key, repr(e)))
            print(f'[{n}/{len(rows)}] !!! FAILED {key}: {e!r}', flush=True)

    put_json(store, 'index.json', sorted(index.values(), key=lambda e: (e['dataset'], e['clip_id'])))
    print(f'index.json: {len(index)} clips')
    if failures:
        print(f'!!! {len(failures)} clip(s) failed:', *failures, sep='\n  ')
        sys.exit(1)


if __name__ == '__main__':
    main()
