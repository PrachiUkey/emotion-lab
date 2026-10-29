// Dev-only media server for EMOTION_LAB_STORE=local:... (R2 mode uses presigned URLs instead).
// Supports Range requests so the browser can seek in mp4s.
import { NextRequest, NextResponse } from 'next/server';
import { isSafeKey, readLocal } from '@/lib/store';

const TYPES: Record<string, string> = { mp4: 'video/mp4', jpg: 'image/jpeg', json: 'application/json' };

export async function GET(req: NextRequest, { params }: { params: { key: string[] } }) {
  const key = params.key.join('/');
  if (!isSafeKey(key)) return NextResponse.json({ error: 'bad key' }, { status: 400 });
  const buf = await readLocal(key);
  if (!buf) return NextResponse.json({ error: 'not found' }, { status: 404 });
  const type = TYPES[key.split('.').pop() ?? ''] ?? 'application/octet-stream';
  const range = req.headers.get('range')?.match(/bytes=(\d*)-(\d*)/);
  if (range) {
    const start = range[1] ? Number(range[1]) : 0;
    const end = range[2] ? Math.min(Number(range[2]), buf.length - 1) : buf.length - 1;
    return new NextResponse(new Uint8Array(buf.subarray(start, end + 1)), {
      status: 206,
      headers: { 'Content-Type': type, 'Content-Range': `bytes ${start}-${end}/${buf.length}`,
                 'Accept-Ranges': 'bytes', 'Content-Length': String(end - start + 1) },
    });
  }
  return new NextResponse(new Uint8Array(buf), {
    headers: { 'Content-Type': type, 'Accept-Ranges': 'bytes', 'Content-Length': String(buf.length) },
  });
}
