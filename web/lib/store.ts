// Server-only access to the emotion-lab store (same layout as emotion_lab/store.py).
import 'server-only';
import { promises as fs } from 'fs';
import path from 'path';
import { GetObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

type Store =
  | { kind: 'local'; root: string }
  | { kind: 'r2'; bucket: string; prefix: string; client: S3Client };

let cached: Store | null = null;

function store(): Store {
  if (cached) return cached;
  const spec = process.env.EMOTION_LAB_STORE;
  if (!spec) throw new Error('EMOTION_LAB_STORE is not set (see web/.env.example)');
  const [kind, rest] = [spec.slice(0, spec.indexOf(':')), spec.slice(spec.indexOf(':') + 1)];
  if (kind === 'local') {
    cached = { kind, root: rest };
  } else if (kind === 'r2') {
    const [bucket, ...prefix] = rest.split('/');
    const { R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY } = process.env;
    if (!R2_ENDPOINT || !R2_ACCESS_KEY_ID || !R2_SECRET_ACCESS_KEY) throw new Error('R2_* env vars are not set');
    cached = {
      kind, bucket, prefix: prefix.join('/'),
      client: new S3Client({
        region: 'auto', endpoint: R2_ENDPOINT, forcePathStyle: true,
        credentials: { accessKeyId: R2_ACCESS_KEY_ID, secretAccessKey: R2_SECRET_ACCESS_KEY },
      }),
    };
  } else {
    throw new Error(`Bad EMOTION_LAB_STORE "${spec}"`);
  }
  return cached;
}

export function isSafeKey(key: string) {
  return /^[\w\-./]+$/.test(key) && !key.split('/').includes('..');
}

export async function getJson<T>(key: string): Promise<T | null> {
  const s = store();
  try {
    if (s.kind === 'local') return JSON.parse(await fs.readFile(path.join(s.root, key), 'utf8'));
    const out = await s.client.send(new GetObjectCommand({ Bucket: s.bucket, Key: `${s.prefix}/${key}` }));
    return JSON.parse(await out.Body!.transformToString());
  } catch (e: unknown) {
    const err = e as { code?: string; name?: string };
    if (err.code === 'ENOENT' || err.name === 'NoSuchKey') return null;
    throw e;
  }
}

// Media (mp4/jpg): R2 -> short-lived presigned URL; local -> served by /api/local-media.
export async function mediaUrl(key: string): Promise<string> {
  const s = store();
  if (s.kind === 'local') return `/api/local-media/${key}`;
  return getSignedUrl(s.client, new GetObjectCommand({ Bucket: s.bucket, Key: `${s.prefix}/${key}` }), { expiresIn: 900 });
}

export async function readLocal(key: string): Promise<Buffer | null> {
  const s = store();
  if (s.kind !== 'local') return null;
  try {
    return await fs.readFile(path.join(s.root, key));
  } catch {
    return null;
  }
}
