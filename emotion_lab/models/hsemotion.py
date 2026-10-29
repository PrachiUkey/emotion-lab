"""HSEmotion (Savchenko) EfficientNet models trained on AffectNet, ONNX weights.
https://github.com/HSE-asavchenko/face-emotion-recognition (Apache-2.0)
"""
import urllib.request  # noqa: F401  hsemotion_onnx calls urllib.request without importing it

import cv2
import numpy as np
import onnxruntime as ort
from hsemotion_onnx.facial_emotions import get_model_path

from emotion_lab.models.base import EmotionModel

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class HSEmotion(EmotionModel):
    def __init__(self, name, variant):
        self.name = name
        self.variant = variant
        self.labels = (['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'] if variant.endswith('_7')
                       else ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'])
        self.size = 224 if '_b0_' in variant else 260
        self.session = ort.InferenceSession(get_model_path(variant),
                                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def _predict(self, crops):
        x = np.stack([(cv2.resize(c, (self.size, self.size)).astype(np.float32) / 255 - MEAN) / STD for c in crops])
        logits = self.session.run(None, {'input': x.transpose(0, 3, 1, 2)})[0][:, :len(self.labels)]
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
