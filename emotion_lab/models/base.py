import numpy as np

from emotion_lab.labels import to_canonical


class EmotionModel:
    """A per-frame facial-expression classifier.

    Subclasses set `name`, `labels` (their own class names, in output order) and implement
    `_predict(crops) -> [N, len(labels)]` probabilities for a list of RGB uint8 face crops.
    """
    name = None
    labels = None
    batch_size = 64

    def _predict(self, crops):
        raise NotImplementedError

    def predict(self, crops):
        """RGB face crops -> [N, len(CANONICAL)] probabilities."""
        if not crops:
            return np.zeros((0, 8), dtype=np.float32)
        out = [self._predict(crops[i:i + self.batch_size]) for i in range(0, len(crops), self.batch_size)]
        return to_canonical(np.concatenate(out), self.labels)
