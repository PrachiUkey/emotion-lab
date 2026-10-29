"""Back up all checkpoints to r2:dump-bucket/emotions_ckpt/ and write a manifest (size + sha256).

    python scripts/upload_checkpoints.py --dry-run     # list + hash only, nothing uploaded
    python scripts/upload_checkpoints.py               # upload, then verify sizes in R2
Needs an R2 key with write access to dump-bucket (.env or R2_* env vars).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotion_lab import r2  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GATED = 'licence-gated upstream (FLAME/BFM/FSGAN terms apply); private bucket only'


def plan(ned_dir):
    """(local path, key under emotions_ckpt/, note)"""
    out = []
    add = lambda p, k, note='': out.append((Path(p), k, note))  # noqa: E731
    for p in sorted((ned_dir / 'manipulator_checkpoints').glob('*.pth')):
        add(p, f'ned/manipulator/{p.name}', 'NED manipulator (public, github.com/foivospar/NED)')
    for d in sorted(ned_dir.glob('checkpoints_*')):
        for p in sorted(d.glob('*.pth')):
            add(p, f"ned/renderers/{d.name.removeprefix('checkpoints_')}/{p.name}", 'NED renderer (public)')
    for d in sorted((ned_dir / 'runs').glob('*/renderer')):
        for p in sorted(d.glob('[0-9]*_net_*.pth')):  # numbered epochs only, not 'latest'
            add(p, f'ned/renderers/{d.parent.name}/{p.name}', 'NED renderer fine-tuned here')
    add(ned_dir / 'DECA/data/deca_model.tar', 'ned/deca/deca_model.tar', 'DECA (public)')
    add(ned_dir / 'DECA/data/generic_model.pkl', 'ned/deca/generic_model.pkl', 'FLAME 2020, ' + GATED)
    add(ned_dir / 'DECA/data/FLAME_albedo_from_BFM.npz', 'ned/deca/FLAME_albedo_from_BFM.npz', 'BFM albedo, ' + GATED)
    seg = 'lfw_figaro_unet_256_2_0_segmentation_v1.pth'
    add(ned_dir / 'preprocessing/segmentation' / seg, f'ned/segmentation/{seg}', 'FSGAN segmenter, ' + GATED)
    for p in sorted((Path.home() / '.hsemotion').glob('*.onnx')):
        add(p, f'classifiers/hsemotion/{p.name}', 'HSEmotion (Apache-2.0)')
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ned_dir', default=str(REPO / 'ned'))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    items = plan(Path(args.ned_dir))
    missing = [str(p) for p, _, _ in items if not p.is_file()]
    if missing:
        sys.exit(f'!!! missing local files: {missing}')
    ned_dir = Path(args.ned_dir).resolve()
    manifest = []
    for p, key, note in items:
        # where fetch_checkpoints.py restores it: repo-relative for NED files, ~-relative for model caches
        p = p.resolve()
        local = (f'ned/{p.relative_to(ned_dir)}' if p.is_relative_to(ned_dir)
                 else f'~/{p.relative_to(Path.home())}')
        # renderers fine-tuned here exist only in R2 (no public source), so a fresh setup may lack them
        manifest.append({'key': f'{r2.CKPT_PREFIX}/{key}', 'path': local, 'size': p.stat().st_size,
                         'sha256': sha256(p), 'required': not local.startswith('ned/runs/'), 'note': note})
        print(f"{manifest[-1]['size'] / 1e6:9.1f} MB  {manifest[-1]['key']}", flush=True)
    total = sum(m['size'] for m in manifest)
    print(f'{len(manifest)} files, {total / 1e9:.2f} GB')
    (REPO / 'configs' / 'checkpoints_manifest.json').write_text(json.dumps(manifest, indent=1))
    if args.dry_run:
        print('dry run: manifest written to configs/checkpoints_manifest.json, nothing uploaded')
        return

    s3 = r2.s3_client()
    for (p, _, _), m in zip(items, manifest):
        s3.upload_file(str(p), r2.DUMP_BUCKET, m['key'], ExtraArgs={'Metadata': {'sha256': m['sha256']}})
        size = s3.head_object(Bucket=r2.DUMP_BUCKET, Key=m['key'])['ContentLength']
        if size != m['size']:
            sys.exit(f"!!! size mismatch after upload: {m['key']} local {m['size']} vs R2 {size}")
        print(f"uploaded + verified {m['key']}", flush=True)
    s3.put_object(Bucket=r2.DUMP_BUCKET, Key=f'{r2.CKPT_PREFIX}/manifest.json',
                  Body=json.dumps(manifest, indent=1).encode(), ContentType='application/json')
    print(f'done: {len(manifest)} files + manifest.json in r2:{r2.DUMP_BUCKET}/{r2.CKPT_PREFIX}/')


if __name__ == '__main__':
    main()
