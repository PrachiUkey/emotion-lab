"""Speech emotion from the clip's audio with emotion2vec+ (the audio emotion encoder C-MET builds on).

The model scores a stretch of speech, so per-frame values come from a sliding window (default 2 s, hop 0.5 s):
each video frame gets the window centred nearest to it. The whole-clip verdict is one pass over all the audio.
emotion2vec also outputs "other" and "<unk>"; those are dropped and the 7 emotions renormalised.
"""
import numpy as np

from emotion_lab.labels import to_canonical
from emotion_lab.models.base import EmotionModel

SR = 16000


class Emotion2Vec(EmotionModel):
    input = 'audio'

    def __init__(self, name, repo='emotion2vec/emotion2vec_plus_large', window=2.0, hop=0.5):
        from funasr import AutoModel
        self.name, self.window, self.hop = name, window, hop
        self.model = AutoModel(model=repo, hub='hf', disable_update=True, log_level='ERROR')
        self.labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    def _score(self, segment):
        res = self.model.generate(segment, granularity='utterance', extract_embedding=False, disable_pbar=True)[0]
        names = [l.split('/')[-1] for l in res['labels']]  # '开心/happy' -> 'happy'
        scores = dict(zip(names, map(float, res['scores'])))
        p = np.array([scores.get(l, 0.0) for l in self.labels], dtype=np.float32)
        return p / p.sum() if p.sum() > 0 else p

    def predict_audio(self, wav, n_frames, fps):
        """wav: mono float32 at 16 kHz. Returns (per-frame canonical probs, whole-clip canonical probs)."""
        win, hop = int(self.window * SR), int(self.hop * SR)
        if len(wav) < SR // 4:
            return [None] * n_frames, None
        starts = list(range(0, max(1, len(wav) - win + 1), hop)) if len(wav) > win else [0]
        windows = np.stack([self._score(wav[s:s + win]) for s in starts])
        centers = (np.array(starts) + min(win, len(wav)) / 2) / SR
        frame_t = (np.arange(n_frames) + 0.5) / fps
        nearest = np.abs(frame_t[:, None] - centers[None, :]).argmin(1)
        per_frame = to_canonical(windows[nearest], self.labels)
        clip = to_canonical(self._score(wav)[None], self.labels)[0]
        return list(per_frame), clip
