'use client';
import { useEffect, useMemo, useRef, useState } from 'react';
import { modelInfo, TERMS } from '@/lib/glossary';
import { EMOTION_COLORS, type PredsDoc } from '@/lib/types';
import Tip from '../../../Tip';

const ABBR: Record<string, string> = {
  neutral: 'neu', happy: 'hap', sad: 'sad', surprised: 'sur', fear: 'fea', disgusted: 'dis', angry: 'ang', contempt: 'con',
};

function rgba(hex: string, a: number) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function top(p: number[], labels: string[]) {
  let b = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i;
  return { label: labels[b], conf: p[b] };
}

type Props = {
  doc: PredsDoc; models: string[]; frame: number; sprite?: string;
  onSeek: (f: number) => void; badge: (m: string) => React.ReactNode;
};

export default function FrameView({ doc, models, frame, sprite, onSeek, badge }: Props) {
  const n = doc.n_frames;
  const auto = Math.max(1, Math.round(n / 60));
  const [step, setStep] = useState(auto);
  const [follow, setFollow] = useState(true);
  const scroller = useRef<HTMLDivElement>(null);
  const cols = useMemo(() => Array.from({ length: Math.ceil(n / step) }, (_, k) => k * step), [n, step]);
  const current = Math.floor(frame / step);
  const CELL = 58;

  // keep the current frame's column in view while the video plays
  useEffect(() => {
    const el = scroller.current;
    if (!follow || !el) return;
    const x = 150 + current * CELL;
    if (x < el.scrollLeft + 150 || x > el.scrollLeft + el.clientWidth - CELL) el.scrollLeft = x - el.clientWidth / 2;
  }, [current, follow]);

  const thumb = (f: number) => {
    const s = doc.sprite;
    if (!sprite || !s) return undefined;
    const k = Math.min(s.count - 1, Math.floor(f / s.step));
    const scale = 40 / s.tile_h;
    return {
      backgroundImage: `url(${sprite})`,
      backgroundSize: `${s.cols * s.tile_w * scale}px auto`,
      backgroundPosition: `-${(k % s.cols) * s.tile_w * scale}px -${Math.floor(k / s.cols) * s.tile_h * scale}px`,
      width: s.tile_w * scale, height: 40,
    };
  };

  const downloadCsv = () => {
    const head = ['frame', 'time_s', 'face_found', 'model', 'top_emotion', 'top_conf', ...doc.labels];
    const lines = [head.join(',')];
    for (let f = 0; f < n; f++) {
      for (const m of models) {
        const p = doc.frames[m][f];
        const t = p ? top(p, doc.labels) : null;
        lines.push([f, (f / doc.fps).toFixed(3), doc.boxes[f] ? 1 : 0, m, t?.label ?? '', t ? t.conf.toFixed(4) : '',
          ...doc.labels.map((_, i) => (p ? p[i].toFixed(4) : ''))].join(','));
      }
    }
    const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }));
    const a = Object.assign(document.createElement('a'), { href: url, download: `${doc.clip_id}_per_frame.csv` });
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="frameview">
      {/* 1. every model, for the frame on screen */}
      <div className="card">
        <div className="fv-head">
          <Tip text={TERMS.thisFrameAll}><b>Frame {frame + 1}</b></Tip>
          <span className="muted">· {(frame / doc.fps).toFixed(2)} s · every model</span>
        </div>
        <div className="fv-now">
          {models.map((m) => {
            const p = doc.frames[m][frame];
            const t = p ? top(p, doc.labels) : null;
            return (
              <div className="fv-row" key={m}>
                <span className="fv-name"><Tip text={modelInfo(m)}>{m}</Tip>{badge(m)}</span>
                <div className="fv-stack" title={p ? doc.labels.map((l, i) => `${l} ${p[i].toFixed(2)}`).join('\n') : ''}>
                  {p ? doc.labels.map((l, i) => p[i] > 0.005 && (
                    <div key={l} style={{ width: `${p[i] * 100}%`, background: EMOTION_COLORS[l] }}>
                      {p[i] > 0.12 ? l : ''}
                    </div>
                  )) : <span className="muted">{doc.model_meta?.[m]?.modality === 'audio' ? 'no sound here' : 'no face here'}</span>}
                </div>
                <span className="chip" style={{ background: t ? EMOTION_COLORS[t.label] : 'var(--line)', color: t ? '#fff' : 'var(--muted)' }}>
                  {t ? `${t.label} ${t.conf.toFixed(2)}` : '—'}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* 2. grid: models x frames */}
      <div className="card">
        <div className="fv-head" style={{ flexWrap: 'wrap' }}>
          <Tip text={TERMS.frameGrid}><b>Frame by frame</b></Tip>
          <label className="muted">show every
            <select value={step} onChange={(e) => setStep(Number(e.target.value))}>
              {Array.from(new Set([1, 2, 5, 10, 25, auto])).sort((a, b) => a - b).map((s) => (
                <option key={s} value={s}>{s === 1 ? 'frame' : `${s}th frame`}{s === auto ? ' (fit)' : ''}</option>
              ))}
            </select>
          </label>
          <label className="muted"><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow video</label>
          <button className="ghost" onClick={downloadCsv}>Download CSV</button>
          <span className="muted" style={{ fontSize: 12 }}>
            {Object.entries(ABBR).map(([k, v]) => <span key={k} style={{ marginRight: 8 }}><b style={{ color: EMOTION_COLORS[k] }}>■</b> {v}={k}</span>)}
          </span>
        </div>
        <div className="fv-grid" ref={scroller}>
          <table>
            <thead>
              <tr>
                <th className="fv-sticky">frame</th>
                {cols.map((f, k) => (
                  <th key={f} className={k === current ? 'fv-cur' : ''} onClick={() => onSeek(f)} title={`frame ${f + 1} · ${(f / doc.fps).toFixed(2)} s`}>
                    {thumb(f) && <div className="fv-thumb" style={thumb(f)} />}
                    <div>{f + 1}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m}>
                  <th className="fv-sticky" title={modelInfo(m)}>{m}</th>
                  {cols.map((f, k) => {
                    const p = doc.frames[m][f];
                    const t = p ? top(p, doc.labels) : null;
                    return (
                      <td key={f} className={k === current ? 'fv-cur' : ''} onClick={() => onSeek(f)}
                          style={t ? { background: rgba(EMOTION_COLORS[t.label], 0.25 + 0.75 * t.conf) } : undefined}
                          title={p ? `frame ${f + 1} · ${m}\n` + doc.labels.map((l, i) => `${l} ${p[i].toFixed(2)}`).join('\n') : `frame ${f + 1} · ${m}: —`}>
                        {t ? <>{ABBR[t.label]}<br /><small>{t.conf.toFixed(2)}</small></> : <span className="muted">—</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
