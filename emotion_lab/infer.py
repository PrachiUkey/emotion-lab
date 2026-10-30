"""Per-frame inference: landmarks -> face crop / aligned frame -> every model -> canonical probabilities."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from emotion_lab.labels import CANONICAL

CHUNK = 64  # frames decoded at a time (full-res frames are large)


def _ffmpeg():
    exe = os.environ.get('FFMPEG') or shutil.which('ffmpeg')
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG = _ffmpeg()


def _summary(per_frame):
    probs = np.array([p for p in per_frame if p is not None])
    mean = probs.mean(0) if len(probs) else np.zeros(len(CANONICAL))
    top = int(mean.argmax())
    return {'label': CANONICAL[top], 'conf': round(float(mean[top]), 4), 'probs': [round(float(v), 4) for v in mean],
            'n_frames': len(probs)}


def predict_clip(clip, models, existing=None):
    """Return the preds document for a clip. Models already in `existing` are kept, not recomputed;
    models restricted to other datasets are skipped."""
    points = clip.points()
    boxes = [None if p is None else _box(p, r) for p, r in zip(points, clip.rows)]
    doc = existing or {'dataset': clip.dataset, 'clip_id': clip.clip_id, 'fragment': clip.fragment,
                       'n_frames': len(clip.rows), 'width': clip.rows[0]['frame_width'],
                       'height': clip.rows[0]['frame_height'], 'labels': CANONICAL, 'boxes': boxes,
                       'frames': {}, 'clip': {}}
    doc['fps'], doc['fps_source'] = clip.fps, clip.fps_source
    if clip.transcript or 'transcript' not in doc:  # keep a stored transcript if this run didn't fetch one
        doc['transcript'] = clip.transcript
    doc['warnings'] = clip.warnings
    doc.setdefault('model_meta', {})
    todo = [m for m in models if m.name not in doc['clip'] and (not m.datasets or clip.dataset in m.datasets)]
    if todo:
        per_frame = {m.name: [None] * len(clip.rows) for m in todo}
        for m in todo:
            if m.input == 'rows':
                for i, p in enumerate(m.predict_rows(clip.rows)):
                    if p is not None and boxes[i] is not None:
                        per_frame[m.name][i] = p
        img_models = [m for m in todo if m.input in ('crop', 'frame')]
        face_idx = [i for i, b in enumerate(boxes) if b is not None]
        for start in range(0, len(face_idx), CHUNK) if img_models else []:
            idx = face_idx[start:start + CHUNK]
            rgbs = [cv2.cvtColor(cv2.imdecode(np.frombuffer(clip.rows[i]['jpeg'], np.uint8), cv2.IMREAD_COLOR),
                                 cv2.COLOR_BGR2RGB) for i in idx]
            crops = [rgb[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2]] for rgb, i in zip(rgbs, idx)]
            for m in img_models:
                probs = m.predict(crops if m.input == 'crop' else list(zip(rgbs, [points[i] for i in idx])))
                for i, p in zip(idx, probs):
                    per_frame[m.name][i] = p
        clip_override = {}
        audio_models = [m for m in todo if m.input == 'audio']
        wav = _load_wav(clip) if audio_models else None
        for m in audio_models:
            if wav is None:
                continue  # no audio for this clip: the model has no predictions here
            per_frame[m.name], clip_probs = m.predict_audio(wav, len(clip.rows), clip.fps)
            if clip_probs is not None:
                clip_override[m.name] = clip_probs
        for m in todo:
            if m.input == 'audio' and m.name not in clip_override:
                continue  # no audio / no speech: leave absent rather than show a made-up prediction
            doc['frames'][m.name] = [None if p is None else [round(float(v), 4) for v in p] for p in per_frame[m.name]]
            doc['clip'][m.name] = _summary(per_frame[m.name])
            if m.name in clip_override:  # audio: one pass over the whole clip beats averaging windows
                probs = clip_override[m.name]
                top = int(np.argmax(probs))
                doc['clip'][m.name].update(label=CANONICAL[top], conf=round(float(probs[top]), 4),
                                           probs=[round(float(v), 4) for v in probs])
            doc['model_meta'][m.name] = m.meta
    doc['face_frac'] = round(sum(b is not None for b in boxes) / len(boxes), 3)
    return doc


def _load_wav(clip, sr=16000):
    """The clip's audio as mono float32 at 16 kHz, or None if it has none."""
    raw = clip.audio()
    if not raw:
        return None
    import io
    import soundfile as sf
    wav, file_sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
    wav = wav.mean(1)
    if file_sr != sr:
        import torch
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), file_sr, sr).numpy()
    return wav


def _box(pts, row):
    from emotion_lab.data import face_box
    return face_box(pts, row['frame_width'], row['frame_height'])


def encode_mp4(clip, max_height=720):
    """JPEG frames (+ the clip's audio, if any) -> H.264/AAC mp4 bytes at the clip's fps, <= max_height."""
    vf = f"scale=-2:'min({max_height},ih)',pad=ceil(iw/2)*2:ceil(ih/2)*2"
    audio = clip.audio()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'out.mp4'
        cmd = [FFMPEG, '-v', 'error', '-y', '-f', 'image2pipe', '-framerate', str(clip.fps), '-c:v', 'mjpeg', '-i', '-']
        if audio:
            (Path(tmp) / 'a.wav').write_bytes(audio)
            # pad audio with silence so -shortest always ends at the last video frame (never drops frames)
            cmd += ['-i', str(Path(tmp) / 'a.wav'), '-map', '0:v', '-map', '1:a', '-af', 'apad',
                    '-c:a', 'aac', '-b:a', '96k', '-shortest']
        cmd += ['-vf', vf, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(out)]
        p = subprocess.run(cmd, input=b''.join(r['jpeg'] for r in clip.rows), capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f'ffmpeg failed for {clip.clip_id}: {p.stderr.decode()[-300:]}')
        return out.read_bytes()


def thumbnail(clip, height=180):
    img = cv2.imdecode(np.frombuffer(clip.rows[len(clip.rows) // 2]['jpeg'], np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * height / h), height))
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def frame_sprite(clip, height=72, max_frames=1000, cols=20):
    """All frames (every `step`-th if the clip is long) as small tiles in one JPEG, for frame-by-frame thumbnails.
    Returns (jpeg bytes, meta) where meta = {step, cols, tile_w, tile_h, count}."""
    n = len(clip.rows)
    step = max(1, -(-n // max_frames))
    idx = list(range(0, n, step))
    first = cv2.imdecode(np.frombuffer(clip.rows[0]['jpeg'], np.uint8), cv2.IMREAD_COLOR)
    tile_h = height
    tile_w = int(round(first.shape[1] * height / first.shape[0])) // 2 * 2
    rows = -(-len(idx) // cols)
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), np.uint8)
    for k, i in enumerate(idx):
        img = cv2.imdecode(np.frombuffer(clip.rows[i]['jpeg'], np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
        r, c = divmod(k, cols)
        sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    jpeg = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    return jpeg, {'step': step, 'cols': cols, 'tile_w': tile_w, 'tile_h': tile_h, 'count': len(idx)}
