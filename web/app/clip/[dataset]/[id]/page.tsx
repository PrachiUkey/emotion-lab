import { notFound } from 'next/navigation';
import { getJson, isSafeKey, mediaUrl } from '@/lib/store';
import type { Gold, PredsDoc } from '@/lib/types';
import Player from './Player';

export const dynamic = 'force-dynamic';

export default async function ClipPage({ params }: { params: { dataset: string; id: string } }) {
  const key = `${params.dataset}/${decodeURIComponent(params.id)}`;
  if (!isSafeKey(key)) notFound();
  const [doc, gold] = await Promise.all([getJson<PredsDoc>(`preds/${key}.json`), getJson<Gold>('gold/labels.json')]);
  if (!doc) notFound();
  const video = await mediaUrl(`clips/${key}.mp4`);
  return (
    <main>
      <h2 style={{ margin: '0 0 12px', fontSize: 16, wordBreak: 'break-all' }}>
        {doc.dataset} · {doc.clip_id}
      </h2>
      <Player doc={doc} video={video} gold={gold?.[key]} />
    </main>
  );
}
