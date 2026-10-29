"""Model registry driven by configs/models.yaml. Adding a model = a `kind` here + an entry in the yaml."""
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'models.yaml'


def _build(name, spec):
    kind = spec['kind']
    if kind == 'hsemotion':
        from emotion_lab.models.hsemotion import HSEmotion
        return HSEmotion(name, spec['variant'])
    if kind == 'hf_image':
        from emotion_lab.models.hf_image import HFImageClassifier
        return HFImageClassifier(name, spec['repo'], spec.get('revision', 'main'))
    raise ValueError(f'{name}: unknown model kind "{kind}"')


def model_specs(config=CONFIG):
    return yaml.safe_load(Path(config).read_text())['models']


def load_models(names=None, config=CONFIG):
    specs = model_specs(config)
    names = names or [n for n, s in specs.items() if s.get('enabled', True)]
    unknown = [n for n in names if n not in specs]
    if unknown:
        raise KeyError(f'Models not in {config}: {unknown}')
    return [_build(n, specs[n]) for n in names]
