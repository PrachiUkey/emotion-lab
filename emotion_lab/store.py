"""Where clips/predictions/eval results live. Same key layout locally and in R2:

    index.json                          clip list + per-model clip summaries (dashboard grid)
    clips/<dataset>/<clip_id>.mp4       H.264 video for playback
    thumbs/<dataset>/<clip_id>.jpg      grid thumbnail
    preds/<dataset>/<clip_id>.json      per-frame boxes + per-model probabilities
    gold/labels.json                    human labels
    eval/leaderboard.json               model comparison

EMOTION_LAB_STORE = local:<dir> (default local:./data_store) | r2:<bucket>/<prefix>
"""
import json
import os
from pathlib import Path

from emotion_lab import r2

REPO = Path(__file__).resolve().parent.parent


class LocalStore:
    def __init__(self, root):
        self.root = Path(root)

    def put_bytes(self, key, data, content_type=None):
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get_bytes(self, key):
        return (self.root / key).read_bytes()

    def exists(self, key):
        return (self.root / key).exists()

    def __repr__(self):
        return f'local:{self.root}'


class R2Store:
    def __init__(self, bucket, prefix):
        self.bucket, self.prefix, self.s3 = bucket, prefix.strip('/'), r2.s3_client()

    def _k(self, key):
        return f'{self.prefix}/{key}' if self.prefix else key

    def put_bytes(self, key, data, content_type=None):
        extra = {'ContentType': content_type} if content_type else {}
        self.s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=data, **extra)

    def get_bytes(self, key):
        return self.s3.get_object(Bucket=self.bucket, Key=self._k(key))['Body'].read()

    def exists(self, key):
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except self.s3.exceptions.ClientError:
            return False

    def __repr__(self):
        return f'r2:{self.bucket}/{self.prefix}'


def open_store(spec=None):
    spec = spec or os.environ.get('EMOTION_LAB_STORE', f'local:{REPO / "data_store"}')
    kind, _, rest = spec.partition(':')
    if kind == 'local':
        return LocalStore(rest)
    if kind == 'r2':
        bucket, _, prefix = rest.partition('/')
        return R2Store(bucket, prefix)
    raise ValueError(f'Bad EMOTION_LAB_STORE "{spec}" (use local:<dir> or r2:<bucket>/<prefix>)')


def put_json(store, key, obj):
    store.put_bytes(key, json.dumps(obj, separators=(',', ':')).encode(), 'application/json')


def get_json(store, key):
    return json.loads(store.get_bytes(key))
