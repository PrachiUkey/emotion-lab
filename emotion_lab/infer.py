"""Per-frame inference: face box from landmarks -> crop -> every model -> canonical probabilities."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from emotion_lab.data import FPS
from emotion_lab.labels import CANONICAL


def _ffmpeg():
    exe = os.environ.get('FFMPEG') or shutil.which('ffmpeg')
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG = _ffmpeg()


def predict_clip(clip, models, existing=None):
    """Return the preds document for a clip. Models already present in `existing` are kept, not recomputed."""
    boxes = clip.boxes()
    doc = existing or {'dataset': clip.dataset, 'clip_id': clip.clip_id, 'fragment': clip.fragment, 'fps': FPS,
                       'n_frames': len(clip.rows), 'width': clip.rows[0]['frame_width'],
                       'height': clip.rows[0]['frame_height'], 'labels': CANONICAL, 'boxes': boxes,
                       'frames': {}, 'clip': {}}
    todo = [m for m in models if m.name not in doc['clip']]
    if not todo:
        return doc
    face_idx = [i for i, b in enumerate(boxes) if b is not None]
    crops = []
    for i, img in enumerate(clip.frames()):
        b = boxes[i]
        if b is not None:
            crops.append(cv2.cvtColor(img[b[1]:b[3], b[0]:b[2]], cv2.COLOR_BGR2RGB))
    for m in todo:
        probs = m.predict(crops) if crops else np.zeros((0, len(CANONICAL)), np.float32)
        per_frame = [None] * len(boxes)
        for i, p in zip(face_idx, probs):
            per_frame[i] = [round(float(v), 4) for v in p]
        doc['frames'][m.name] = per_frame
        mean = probs.mean(0) if len(probs) else np.zeros(len(CANONICAL))
        top = int(mean.argmax())
        doc['clip'][m.name] = {'label': CANONICAL[top], 'conf': round(float(mean[top]), 4),
                               'probs': [round(float(v), 4) for v in mean]}
    doc['face_frac'] = round(len(face_idx) / len(boxes), 3)
    return doc


def encode_mp4(clip, max_height=720):
    """JPEG frames -> H.264 mp4 bytes (fps assumed), downscaled to <= max_height."""
    vf = f"scale=-2:'min({max_height},ih)',pad=ceil(iw/2)*2:ceil(ih/2)*2"
    # +faststart (moov atom first, for streaming in browsers) needs a seekable output, so go via a temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
        p = subprocess.run([FFMPEG, '-v', 'error', '-y', '-f', 'image2pipe', '-framerate', str(FPS), '-c:v', 'mjpeg',
                            '-i', '-', '-vf', vf, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', tmp.name],
                           input=b''.join(r['jpeg'] for r in clip.rows), capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f'ffmpeg failed for {clip.clip_id}: {p.stderr.decode()[-300:]}')
        return Path(tmp.name).read_bytes()


def thumbnail(clip, height=180):
    img = next(clip.frames()) if len(clip.rows) < 2 else cv2.imdecode(
        np.frombuffer(clip.rows[len(clip.rows) // 2]['jpeg'], np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * height / h), height))
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
