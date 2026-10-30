import numpy as np

from emotion_lab.labels import CANONICAL, to_canonical


class EmotionModel:
    """A per-frame facial-expression classifier.

    Subclasses set `name`, `labels` (their own class names, in output order) and implement one of:
      input = 'crop'  -> `_predict(crops)`: RGB face crops from the landmark box
      input = 'frame' -> `_predict(items)`: (RGB full frame, 68x2 landmarks) pairs, for models that align faces themselves
      input = 'rows'  -> `predict_rows(rows)`: raw dataset rows (e.g. precomputed scores); returns per-frame
                         canonical probabilities or None
    `datasets`: if set, the model only runs on clips from these datasets.
    """
    name = None
    labels = None
    input = 'crop'
    datasets = None
    batch_size = 64

    def _predict(self, items):
        raise NotImplementedError

    def predict(self, items):
        """Model inputs -> [N, len(CANONICAL)] probabilities."""
        if not items:
            return np.zeros((0, len(CANONICAL)), dtype=np.float32)
        out = [self._predict(items[i:i + self.batch_size]) for i in range(0, len(items), self.batch_size)]
        return to_canonical(np.concatenate(out), self.labels)

    def classes(self):
        """Canonical emotions this model can output (others always get probability 0)."""
        from emotion_lab.labels import canonical_name
        return [canonical_name(l) for l in self.labels]
