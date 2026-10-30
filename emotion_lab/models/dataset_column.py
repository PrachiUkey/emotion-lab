"""Emotion scores that ship with a dataset (computed by its authors), shown like any other model.

Seamless Interaction's frames_full.lance has `emotion_scores`: 8 logits per frame from Meta's
"Imitator" model; all-zero rows mean "not computed".
"""
import numpy as np

from emotion_lab.labels import CANONICAL, to_canonical
from emotion_lab.models.base import EmotionModel


class DatasetColumn(EmotionModel):
    input = 'rows'

    def __init__(self, name, column, labels, datasets, logits=True):
        self.name, self.column, self.labels, self.datasets, self.logits = name, column, labels, datasets, logits

    def predict_rows(self, rows):
        out = []
        for r in rows:
            v = r.get(self.column)
            a = None if v is None else np.asarray(v, dtype=np.float32)
            if a is None or not a.any():
                out.append(None)
                continue
            if self.logits:
                a = np.exp(a - a.max())
                a /= a.sum()
            out.append(to_canonical(a[None], self.labels)[0])
        return out

    def predict(self, items):  # not crop-based
        return np.zeros((0, len(CANONICAL)), dtype=np.float32)
