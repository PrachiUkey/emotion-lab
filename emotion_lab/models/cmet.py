"""The emotion classifier C-MET uses to score its generated videos (evaluation/Emotion-FAN in
github.com/ChanHyeok-Choi/C-MET, Apache-2.0): ResNet18 initialised on FER+, fine-tuned on MEAD or CREMA-D,
applied per frame. It has no "neutral" class (C-MET's own evaluation never runs it on neutral clips).

Inputs reproduce their preprocessing: dlib 5-point face alignment, get_face_chip(size=224, padding=0.25),
then ImageNet normalisation. We locate the face from the dataset's landmarks instead of dlib's detector.
"""
from pathlib import Path

import numpy as np
import torch
import torchvision

from emotion_lab.models.base import EmotionModel

CACHE = Path.home() / '.cache' / 'emotion-lab' / 'cmet'
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LABELS = {  # output order, from emotion-fan.py / emotion-fan_crema.py
    'mead': ['happy', 'angry', 'disgusted', 'fear', 'sad', 'contempt', 'surprised'],
    'crema_d': ['happy', 'angry', 'disgusted', 'fear', 'sad'],
}


class CMetEmotionFAN(EmotionModel):
    input = 'frame'

    def __init__(self, name, variant, checkpoint, device='cuda'):
        import dlib
        self.name = name
        self.labels = LABELS[variant]
        self.device = device if torch.cuda.is_available() else 'cpu'
        ck = torch.load(CACHE / checkpoint, map_location='cpu', weights_only=True)
        sd = {k.replace('base_model.', '').replace('module.', ''): v for k, v in ck['state_dict'].items()}
        self.model = torchvision.models.resnet18(num_classes=len(self.labels))
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device).eval()
        self.dlib = dlib
        self.shape_predictor = dlib.shape_predictor(str(CACHE / 'shape_predictor_5_face_landmarks.dat'))

    def align(self, rgb, pts68):
        """dlib 5-point alignment inside the landmark box, same chip as Emotion-FAN's face_align_cuda.py."""
        x0, y0 = pts68.min(0)
        x1, y1 = pts68.max(0)
        rect = self.dlib.rectangle(int(x0), int(y0), int(x1), int(y1))
        shape = self.shape_predictor(rgb, rect)
        return self.dlib.get_face_chip(rgb, shape, size=224, padding=0.25)

    @torch.no_grad()
    def _predict(self, items):
        chips = np.stack([self.align(rgb, pts) for rgb, pts in items]).astype(np.float32) / 255
        x = torch.from_numpy(((chips - MEAN) / STD).transpose(0, 3, 1, 2)).to(self.device)
        return self.model(x).softmax(-1).float().cpu().numpy()
