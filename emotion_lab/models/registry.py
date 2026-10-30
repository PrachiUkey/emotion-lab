"""Model registry driven by configs/models.yaml. Adding a model = a `kind` here + an entry in the yaml."""
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'models.yaml'


def _build(name, spec):
    kind = spec['kind']
    if kind == 'hsemotion':
        from emotion_lab.models.hsemotion import HSEmotion
        model = HSEmotion(name, spec['variant'])
    elif kind == 'hf_image':
        from emotion_lab.models.hf_image import HFImageClassifier
        model = HFImageClassifier(name, spec['repo'], spec.get('revision', 'main'))
    elif kind == 'cmet_fan':
        from emotion_lab.models.cmet import CMetEmotionFAN
        model = CMetEmotionFAN(name, spec['variant'], spec['checkpoint'])
    elif kind == 'audio_emotion2vec':
        from emotion_lab.models.audio_emotion2vec import Emotion2Vec
        model = Emotion2Vec(name, spec.get('repo', 'emotion2vec/emotion2vec_plus_large'), spec.get('window', 2.0),
                            spec.get('hop', 0.5), spec.get('silence_db', -55.0), spec.get('min_voiced', 0.2))
    elif kind == 'dataset_column':
        from emotion_lab.models.dataset_column import DatasetColumn
        model = DatasetColumn(name, spec['column'], spec['labels'], spec['datasets'], spec.get('logits', True))
    else:
        raise ValueError(f'{name}: unknown model kind "{kind}"')
    model.datasets = spec.get('datasets', model.datasets)
    # shown in the dashboard: which emotions it can say at all, and where it was trained
    model.meta = {'classes': model.classes(), 'trained_on': spec.get('trained_on', ''), 'kind': kind,
                  'modality': 'audio' if model.input == 'audio' else 'face'}
    return model


def model_specs(config=CONFIG):
    return yaml.safe_load(Path(config).read_text())['models']


def load_models(names=None, config=CONFIG):
    specs = model_specs(config)
    names = names or [n for n, s in specs.items() if s.get('enabled', True)]
    unknown = [n for n in names if n not in specs]
    if unknown:
        raise KeyError(f'Models not in {config}: {unknown}')
    return [_build(n, specs[n]) for n in names]
