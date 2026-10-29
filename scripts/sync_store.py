"""Copy the local store (data_store/: index, preds, clips, thumbs, eval, gold) to R2 for the Vercel dashboard.

    python scripts/sync_store.py --dry-run     # show what would be uploaded
    python scripts/sync_store.py               # upload (needs a write-enabled R2 key)
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from emotion_lab import r2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(REPO / 'data_store'))
    ap.add_argument('--dest', default=f'{r2.DUMP_BUCKET}/{r2.LAB_PREFIX}')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if not shutil.which('rclone'):
        sys.exit('rclone not found (https://rclone.org/install/)')
    if not (Path(args.src) / 'index.json').is_file():
        sys.exit(f'{args.src}/index.json missing: run scripts/run_models.py first')
    cmd = ['rclone', 'copy', args.src, f'r2:{args.dest}', '--checksum', '--transfers', '16', '--stats-one-line', '-v']
    if args.dry_run:
        cmd.append('--dry-run')
    res = subprocess.run(cmd, env=r2.rclone_env())  # remote defined via env vars, no config file on disk
    if res.returncode != 0:
        sys.exit(f'!!! rclone failed (exit {res.returncode}). AccessDenied means the R2 key cannot write to '
                 f'{args.dest.split("/")[0]}.')
    print('dry run only, nothing uploaded' if args.dry_run else f'synced {args.src} -> r2:{args.dest}')


if __name__ == '__main__':
    main()
