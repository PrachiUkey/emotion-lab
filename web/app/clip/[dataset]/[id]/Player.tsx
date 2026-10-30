'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { modelInfo, TERMS } from '@/lib/glossary';
import { EMOTION_COLORS, type Gold, type PredsDoc } from '@/lib/types';
import Tip from '../../../Tip';

type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
};

function top(probs: number[], labels: string[]) {
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { label: labels[best], conf: probs[best] };
}

function label(ctx: CanvasRenderingContext2D, text: string, color: string, x: number, y: number) {
  const tw = ctx.measureText(text).width + 10;
  ctx.fillStyle = color; ctx.fillRect(x, y, tw, 20);
  ctx.fillStyle = '#fff'; ctx.fillText(text, x + 5, y + 14);
}

export default function Player({ doc, video, gold }: { doc: PredsDoc; video: string; gold?: Gold[string] }) {
  const meta = doc.model_meta ?? {};
  const isAudio = (m: string) => meta[m]?.modality === 'audio';
  const classesOf = (m: string) => meta[m]?.classes ?? doc.labels;
  // face models first, then voice models
  const models = Object.keys(doc.frames).sort((a, b) => Number(isAudio(a)) - Number(isAudio(b)) || a.localeCompare(b));
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
      const tick = (_: number, m: { mediaTime: number }) => {
        if (stop) return;
        setFrame(frameAt(m.mediaTime));
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

  // Face models: box + label(s) on the face. Voice models: label in the top-left corner (no face needed).
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
    ctx.font = '600 13px system-ui';
    const shown = model === 'all' ? models : [model];
    const tops = (ms: string[]) => ms.map((m) => ({ m, p: doc.frames[m][frame] })).filter((r) => r.p)
      .map((r) => ({ m: r.m, ...top(r.p!, doc.labels) }));

    const voice = tops(shown.filter(isAudio));
    voice.forEach((t, i) => label(ctx, `🔊 ${model === 'all' ? `${t.m}: ` : 'voice: '}${t.label} ${t.conf.toFixed(2)}`,
      EMOTION_COLORS[t.label], 8, 8 + 22 * i));

    const faceModels = shown.filter((m) => !isAudio(m));
    if (!faceModels.length) return;
    const box = doc.boxes[frame];
    if (!box) {
      label(ctx, 'no face', 'rgba(0,0,0,.6)', 8, 8 + 22 * voice.length);
      return;
    }
    const sx = w / doc.width, sy = h / doc.height;
    const [x0, y0, x1, y1] = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy];
    const face = tops(faceModels);
    ctx.lineWidth = 3;
    ctx.strokeStyle = face[0] ? EMOTION_COLORS[face[0].label] : '#fff';
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    face.forEach((t, i) => label(ctx, `${model === 'all' ? `${t.m}: ` : ''}${t.label} ${t.conf.toFixed(2)}`,
      EMOTION_COLORS[t.label], x0, Math.max(0, y0 - 22 * (face.length - i))));
  }, [frame, model, doc, models]); // eslint-disable-line react-hooks/exhaustive-deps

  const seek = (f: number) => { if (videoRef.current) videoRef.current.currentTime = (f + 0.5) / doc.fps; };
  const barModel = model === 'all' ? models[0] : model;
  const probs = doc.frames[barModel]?.[frame];
  const can = new Set(classesOf(barModel));
  const t = (frame + 0.5) / doc.fps;
  const clipEnd = doc.n_frames / doc.fps;  // transcripts can run past the clip; never show words after it ends
  const words = (doc.transcript ?? []).filter((w) => w.start < clipEnd && w.end >= t - 2.5 && w.start <= t + 2.5);
  const fpsAssumed = doc.fps_source !== 'dataset';
  const badge = (m: string) => (
    <>
      {isAudio(m) && <span className="badge">voice</span>}
      {!isAudio(m) && !classesOf(m).includes('neutral') && <span className="badge warn">no neutral</span>}
    </>
  );

  return (
    <div className="player">
      <div>
        <div className="stage">
          {/* the overlay canvas sits exactly on the video box, so portrait clips can be height-capped */}
          <div className="frame">
            <video ref={videoRef} src={video} controls playsInline preload="auto" />
            <canvas ref={canvasRef} />
          </div>
        </div>
        {doc.transcript && doc.transcript.length > 0 && (
          <div className="caption">
            <Tip text={TERMS.transcript}><span className="muted">Said: </span></Tip>
            {words.length ? words.map((w, i) => (
              <span key={i} className={w.start <= t && t <= w.end ? 'now' : ''}>{w.word} </span>
            )) : <span className="muted">…</span>}
          </div>
        )}
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
          <Tip text={TERMS.timeline}>Timeline</Tip>
          {models.map((m, i) => <Tip key={m} text={modelInfo(m)}>row {i + 1}: {m}{isAudio(m) ? ' (voice)' : ''}</Tip>)}
          <span>· frame {frame + 1}/{doc.n_frames}</span>
          {fpsAssumed ? <Tip text={TERMS.fps}>@ {doc.fps} fps (assumed)</Tip> : <span>@ {doc.fps.toFixed(2)} fps</span>}
        </div>
        {doc.warnings && doc.warnings.length > 0 && (
          <div className="muted" style={{ fontSize: 12 }}>
            <Tip text={TERMS.dataWarning}>⚠ data note:</Tip> {doc.warnings.join('; ')}
          </div>
        )}
      </div>

      <aside className="side">
        <div className="card">
          <div className="muted" style={{ marginBottom: 6 }}><Tip text={TERMS.overlay}>Overlay</Tip></div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {[...models, 'all'].map((m) => (
              <Tip key={m} interactive side="right" text={m === 'all' ? TERMS.all : modelInfo(m)}>
                <button className={`ghost ${model === m ? 'on' : ''}`} onClick={() => setModel(m)}>{m}{m !== 'all' && badge(m)}</button>
              </Tip>
            ))}
          </div>
        </div>
        {gold && (
          <div className="card">
            <div className="muted"><Tip text={TERMS.humanLabel}>Human label</Tip></div>
            <div style={{ fontWeight: 600, color: EMOTION_COLORS[gold.emotion] }}>
              {gold.emotion}{' '}
              {gold.confidence && (
                <Tip side="right" text={TERMS.confidence[gold.confidence] ?? 'How sure the person labelling was.'}>
                  <span className="muted">({gold.confidence})</span>
                </Tip>
              )}
            </div>
            {gold.notes && (
              <div className="muted" style={{ fontSize: 12 }}><Tip side="right" text={TERMS.notes}>{gold.notes}</Tip></div>
            )}
          </div>
        )}
        <div className="card">
          <div className="muted" style={{ marginBottom: 6 }}>
            <Tip text={isAudio(barModel) ? TERMS.voiceFrame : TERMS.frame}>Frame {frame + 1} · {barModel}</Tip>
          </div>
          {probs ? (
            <div className="bars">
              {doc.labels.map((l, i) => can.has(l) ? (
                <div className="bar" key={l}>
                  <span>{l}</span>
                  <div className="track"><div className="fill" style={{ width: `${probs[i] * 100}%`, background: EMOTION_COLORS[l] }} /></div>
                  <span>{probs[i].toFixed(2)}</span>
                </div>
              ) : (
                <div className="bar off" key={l}>
                  <span>{l}</span>
                  <Tip side="right" text={TERMS.cannotSay}><span className="muted" style={{ fontSize: 11 }}>can&apos;t say</span></Tip>
                  <span />
                </div>
              ))}
            </div>
          ) : <div className="muted">{isAudio(barModel) ? 'No audio score here.' : 'No face in this frame.'}</div>}
        </div>
        <div className="card">
          <div className="muted" style={{ marginBottom: 6 }}>
            <Tip text={TERMS.wholeClip}>Whole clip</Tip>
          </div>
          {models.map((m) => (
            <div key={m} className="bar" style={{ gridTemplateColumns: '1fr auto' }}>
              <span><Tip text={modelInfo(m)}>{m}</Tip>{badge(m)}</span>
              <span className="chip" style={{ background: EMOTION_COLORS[doc.clip[m].label] }}>
                {doc.clip[m].label} {doc.clip[m].conf.toFixed(2)}
              </span>
            </div>
          ))}
          <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
            <Tip text={TERMS.faceVisible}>Face visible in {(doc.face_frac * 100).toFixed(0)}% of frames</Tip>
          </div>
        </div>
      </aside>
    </div>
  );
}
