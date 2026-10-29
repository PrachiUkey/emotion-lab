"""Canonical emotion label set shared by all models, gold labels and the dashboard.

The first 7 are NED's emotions (the editing pipeline's targets); 'contempt' is kept separate because
AffectNet-trained models predict it but FER2013 models and NED do not.
"""
import numpy as np

CANONICAL = ['neutral', 'happy', 'sad', 'surprised', 'fear', 'disgusted', 'angry', 'contempt']

# Model-specific class names -> canonical
ALIASES = {
    'neutral': 'neutral', 'happiness': 'happy', 'happy': 'happy', 'sadness': 'sad', 'sad': 'sad',
    'surprise': 'surprised', 'surprised': 'surprised', 'fear': 'fear', 'disgust': 'disgusted',
    'disgusted': 'disgusted', 'anger': 'angry', 'angry': 'angry', 'contempt': 'contempt',
}


def canonical_name(name):
    key = name.strip().lower()
    if key not in ALIASES:
        raise KeyError(f'Unknown emotion label "{name}"; add it to emotion_lab.labels.ALIASES')
    return ALIASES[key]


def to_canonical(probs, model_labels):
    """[N, len(model_labels)] -> [N, len(CANONICAL)]; classes a model lacks get probability 0."""
    probs = np.asarray(probs, dtype=np.float32)
    out = np.zeros((probs.shape[0], len(CANONICAL)), dtype=np.float32)
    for j, name in enumerate(model_labels):
        out[:, CANONICAL.index(canonical_name(name))] += probs[:, j]
    return out
