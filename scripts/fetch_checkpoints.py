"""Restore every checkpoint listed in configs/checkpoints_manifest.json from r2:dump-bucket/emotions_ckpt/.

    python scripts/fetch_checkpoints.py            # download missing/corrupt files, verify sha256
    python scripts/fetch_checkpoints.py --check    # only verify what is on disk
Exits non-zero (listing every problem) if anything is missing or fails verification.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from emotion_lab import r2  # noqa: E402

MANIFEST = REPO / 'configs' / 'checkpoints_manifest.json'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def local_path(entry):
    p = entry['path']
    return Path(os.path.expanduser(p)) if p.startswith('~') else REPO / p


def ok(entry):
    p = local_path(entry)
    return p.is_file() and p.stat().st_size == entry['size'] and sha256(p) == entry['sha256']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='verify only, no downloads')
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    s3 = None if args.check else r2.s3_client()
    problems, optional_missing = [], []
    for e in manifest:
        dest = local_path(e)
        if ok(e):
            print(f'ok        {e["path"]}')
            continue
        if args.check:
            (problems if e.get('required', True) else optional_missing).append(
                f'{e["path"]}: missing or checksum mismatch')
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + '.part')
        try:
            print(f'download  {e["key"]} ({e["size"] / 1e6:.0f} MB)', flush=True)
            s3.download_file(r2.DUMP_BUCKET, e['key'], str(part))
        except Exception as ex:  # report all failures at the end
            (problems if e.get('required', True) else optional_missing).append(
                f'{e["key"]}: download failed: {ex!r}')
            part.unlink(missing_ok=True)
            continue
        if sha256(part) != e['sha256']:
            problems.append(f'{e["key"]}: sha256 mismatch after download')
            part.unlink()
            continue
        part.replace(dest)
        print(f'verified  {e["path"]}')
    if optional_missing:
        print(f'\n!!! {len(optional_missing)} optional checkpoint(s) unavailable (renderers trained on another '
              f'machine; only in R2):', *optional_missing, sep='\n  ')
    if problems:
        print(f'\n!!! {len(problems)} required checkpoint problem(s):', *problems, sep='\n  ')
        sys.exit(1)
    print(f'{len(manifest) - len(optional_missing)}/{len(manifest)} checkpoints present and verified')


if __name__ == '__main__':
    main()
