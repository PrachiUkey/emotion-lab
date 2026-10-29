import { getJson, mediaUrl } from '@/lib/store';
import type { Gold, IndexEntry } from '@/lib/types';
import ClipGrid from './ClipGrid';

export const dynamic = 'force-dynamic';

export default async function Home() {
  const [index, gold] = await Promise.all([getJson<IndexEntry[]>('index.json'), getJson<Gold>('gold/labels.json')]);
  if (!index) {
    return <main><p className="error">No index.json in the store yet. Run scripts/run_models.py first.</p></main>;
  }
  const thumbs = await Promise.all(index.map((e) => mediaUrl(`thumbs/${e.dataset}/${e.clip_id}.jpg`)));
  const models = Array.from(new Set(index.flatMap((e) => Object.keys(e.models)))).sort();
  return (
    <main>
      <ClipGrid entries={index.map((e, i) => ({ ...e, thumb: thumbs[i], gold: gold?.[`${e.dataset}/${e.clip_id}`]?.emotion }))}
                models={models} />
    </main>
  );
}
