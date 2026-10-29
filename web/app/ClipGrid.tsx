'use client';
import Link from 'next/link';
import { useMemo, useState } from 'react';
import { EMOTION_COLORS, type IndexEntry } from '@/lib/types';

type Entry = IndexEntry & { thumb: string; gold?: string };

export default function ClipGrid({ entries, models }: { entries: Entry[]; models: string[] }) {
  const [dataset, setDataset] = useState('all');
  const [model, setModel] = useState(models[0] ?? '');
  const [emotion, setEmotion] = useState('all');
  const [view, setView] = useState('all');

  const datasets = useMemo(() => Array.from(new Set(entries.map((e) => e.dataset))).sort(), [entries]);
  const shown = entries.filter((e) => {
    if (dataset !== 'all' && e.dataset !== dataset) return false;
    if (emotion !== 'all' && e.models[model]?.label !== emotion) return false;
    const labels = new Set(Object.values(e.models).map((m) => m.label));
    if (view === 'disagree' && labels.size < 2) return false;
    if (view === 'gold' && !e.gold) return false;
    if (view === 'wrong' && (!e.gold || e.models[model]?.label === e.gold)) return false;
    return true;
  });

  return (
    <>
      <div className="filters">
        <select value={dataset} onChange={(e) => setDataset(e.target.value)} aria-label="Dataset">
          <option value="all">All datasets</option>
          {datasets.map((d) => <option key={d}>{d}</option>)}
        </select>
        <select value={model} onChange={(e) => setModel(e.target.value)} aria-label="Model">
          {models.map((m) => <option key={m}>{m}</option>)}
        </select>
        <select value={emotion} onChange={(e) => setEmotion(e.target.value)} aria-label="Emotion">
          <option value="all">Any emotion</option>
          {Object.keys(EMOTION_COLORS).map((k) => <option key={k}>{k}</option>)}
        </select>
        <select value={view} onChange={(e) => setView(e.target.value)} aria-label="View">
          <option value="all">All clips</option>
          <option value="disagree">Models disagree</option>
          <option value="gold">Has human label</option>
          <option value="wrong">Selected model ≠ human label</option>
        </select>
        <span className="muted">{shown.length} / {entries.length} clips</span>
      </div>
      <div className="grid">
        {shown.map((e) => (
          <Link key={`${e.dataset}/${e.clip_id}`} href={`/clip/${e.dataset}/${encodeURIComponent(e.clip_id)}`} className="tile">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={e.thumb} alt="" loading="lazy" />
            <div className="meta">
              <div className="id" title={e.clip_id}>{e.dataset} · {e.clip_id}</div>
              <div className="chips">
                {e.gold && <span className="chip gold" style={{ color: EMOTION_COLORS[e.gold] }}>human: {e.gold}</span>}
                {models.map((m) => e.models[m] && (
                  <span key={m} className="chip" style={{ background: EMOTION_COLORS[e.models[m].label], opacity: m === model ? 1 : 0.55 }}
                        title={m}>
                    {m}: {e.models[m].label} {e.models[m].conf.toFixed(2)}
                  </span>
                ))}
              </div>
            </div>
          </Link>
        ))}
      </div>
    </>
  );
}
