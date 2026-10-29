'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { EMOTION_COLORS, type Gold, type PredsDoc } from '@/lib/types';

type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
};

function top(probs: number[], labels: string[]) {
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { label: labels[best], conf: probs[best] };
}

export default function Player({ doc, video, gold }: { doc: PredsDoc; video: string; gold?: Gold[string] }) {
  const models = Object.keys(doc.frames).sort();
  const [model, setModel] = useState<string>(models[0] ?? 'all');
  const [frame, setFrame] = useState(0);
  const videoRef = useRef<VideoWithRVFC>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const frameAt = useCallback((t: number) => Math.min(doc.n_frames - 1, Math.max(0, Math.floor(t * doc.fps + 1e-3))),
    [doc.n_frames, doc.fps]);

  // Track the displayed frame: requestVideoFrameCallback when available (frame-accurate), else rAF polling.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    let stop = false;
    if (v.requestVideoFrameCallback) {
      const tick = (_: number, meta: { mediaTime: number }) => {
        if (stop) return;
        setFrame(frameAt(meta.mediaTime));
        v.requestVideoFrameCallback!(tick);
      };
      v.requestVideoFrameCallback(tick);
    } else {
      const loop = () => { if (stop) return; setFrame(frameAt(v.currentTime)); requestAnimationFrame(loop); };
      requestAnimationFrame(loop);
    }
    const onSeek = () => setFrame(frameAt(v.currentTime));
    v.addEventListener('seeked', onSeek);
    return () => { stop = true; v.removeEventListener('seeked', onSeek); };
  }, [frameAt]);

  // Draw box + label(s) for the current frame.
  useEffect(() => {
    const c = canvasRef.current;
    const v = videoRef.current;
    if (!c || !v) return;
    const dpr = window.devicePixelRatio || 1;
    const w = v.clientWidth, h = v.clientHeight;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    const ctx = c.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const box = doc.boxes[frame];
    if (!box) {
      ctx.fillStyle = 'rgba(0,0,0,.6)'; ctx.fillRect(8, 8, 120, 22);
      ctx.fillStyle = '#fff'; ctx.font = '12px system-ui'; ctx.fillText('no face', 16, 23);
      return;
    }
    const sx = w / doc.width, sy = h / doc.height;
    const [x0, y0, x1, y1] = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy];
    const shown = model === 'all' ? models : [model];
    const tops = shown.map((m) => ({ m, p: doc.frames[m][frame] })).filter((r) => r.p).map((r) => ({ m: r.m, ...top(r.p!, doc.labels) }));
    const main = tops[0];
    ctx.lineWidth = 3;
    ctx.strokeStyle = main ? EMOTION_COLORS[main.label] : '#fff';
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.font = '600 13px system-ui';
    tops.forEach((t, i) => {
      const text = `${model === 'all' ? `${t.m}: ` : ''}${t.label} ${t.conf.toFixed(2)}`;
      const tw = ctx.measureText(text).width + 10;
      const ty = Math.max(0, y0 - 22 * (tops.length - i));
      ctx.fillStyle = EMOTION_COLORS[t.label]; ctx.fillRect(x0, ty, tw, 20);
      ctx.fillStyle = '#fff'; ctx.fillText(text, x0 + 5, ty + 14);
    });
  }, [frame, model, doc, models]);

  const seek = (f: number) => { if (videoRef.current) videoRef.current.currentTime = (f + 0.5) / doc.fps; };
  const barModel = model === 'all' ? models[0] : model;
  const probs = doc.frames[barModel]?.[frame];

  return (
    <div className="player">
      <div>
        <div className="stage">
          <video ref={videoRef} src={video} controls playsInline preload="auto" />
          <canvas ref={canvasRef} />
        </div>
        <svg className="timeline" viewBox={`0 0 ${doc.n_frames} ${models.length * 12 + 4}`} preserveAspectRatio="none"
             style={{ height: models.length * 22 + 8 }}
             onClick={(e) => {
               const r = e.currentTarget.getBoundingClientRect();
               seek(Math.floor(((e.clientX - r.left) / r.width) * doc.n_frames));
             }}>
          {models.map((m, row) => doc.frames[m].map((p, i) => p && (
            <rect key={`${m}-${i}`} x={i} y={row * 12 + 2} width={1.05} height={10} fill={EMOTION_COLORS[top(p, doc.labels).label]} />
          )))}
          <rect x={frame} y={0} width={Math.max(1, doc.n_frames / 300)} height={models.length * 12 + 4} fill="currentColor" />
        </svg>
        <div className="muted" style={{ fontSize: 12, display: 'flex', gap: 14, flexWrap: 'wrap' }}>
          {models.map((m) => <span key={m}>row {models.indexOf(m) + 1}: {m}</span>)}
          <span>· click to seek · frame {frame + 1}/{doc.n_frames} @ {doc.fps} fps (assumed)</span>
        </div>
      </div>

      <aside className="side">
        <div className="card">
          <div className="muted" style={{ marginBottom: 6 }}>Overlay</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {[...models, 'all'].map((m) => (
              <button key={m} className={`ghost ${model === m ? 'on' : ''}`} onClick={() => setModel(m)}>{m}</button>
            ))}
          </div>
        </div>
        {gold && (
          <div className="card">
            <div className="muted">Human label</div>
            <div style={{ fontWeight: 600, color: EMOTION_COLORS[gold.emotion] }}>{gold.emotion} <span className="muted">({gold.confidence})</span></div>
            {gold.notes && <div className="muted" style={{ fontSize: 12 }}>{gold.notes}</div>}
          </div>
        )}
        <div className="card">
          <div className="muted" style={{ marginBottom: 6 }}>Frame {frame + 1} · {barModel}</div>
          {probs ? (
            <div className="bars">
              {doc.labels.map((l, i) => (
                <div className="bar" key={l}>
                  <span>{l}</span>
                  <div className="track"><div className="fill" style={{ width: `${probs[i] * 100}%`, background: EMOTION_COLORS[l] }} /></div>
                  <span>{probs[i].toFixed(2)}</span>
                </div>
              ))}
            </div>
          ) : <div className="muted">No face in this frame.</div>}
        </div>
        <div className="card">
          <div className="muted" style={{ marginBottom: 6 }}>Whole clip (mean over face frames)</div>
          {models.map((m) => (
            <div key={m} className="bar" style={{ gridTemplateColumns: '1fr auto' }}>
              <span>{m}</span>
              <span className="chip" style={{ background: EMOTION_COLORS[doc.clip[m].label] }}>
                {doc.clip[m].label} {doc.clip[m].conf.toFixed(2)}
              </span>
            </div>
          ))}
          <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>Face visible in {(doc.face_frac * 100).toFixed(0)}% of frames</div>
        </div>
      </aside>
    </div>
  );
}
