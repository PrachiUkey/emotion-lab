"""Any Hugging Face image-classification checkpoint whose labels are emotion names
(e.g. trpakov/vit-face-expression, ViT fine-tuned on FER2013)."""
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from emotion_lab.models.base import EmotionModel


class HFImageClassifier(EmotionModel):
    batch_size = 128

    def __init__(self, name, repo, revision='main', device='cuda'):
        self.name = name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.processor = AutoImageProcessor.from_pretrained(repo, revision=revision)
        self.model = AutoModelForImageClassification.from_pretrained(repo, revision=revision).to(self.device).eval()
        cfg = self.model.config.id2label
        self.labels = [cfg[i] for i in range(len(cfg))]

    @torch.no_grad()
    def _predict(self, crops):
        inputs = self.processor(images=[Image.fromarray(c) for c in crops], return_tensors='pt').to(self.device)
        return self.model(**inputs).logits.softmax(-1).float().cpu().numpy()
