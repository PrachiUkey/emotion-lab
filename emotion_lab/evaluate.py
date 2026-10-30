"""Compare models against human gold labels (clip level) and against each other."""
from itertools import combinations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from emotion_lab.labels import CANONICAL


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def flip_rate(frame_probs):
    """Fraction of consecutive face frames whose top label changes (temporal jitter)."""
    tops = [int(np.argmax(p)) for p in frame_probs if p is not None]
    if len(tops) < 2:
        return 0.0
    return float(np.mean([a != b for a, b in zip(tops, tops[1:])]))


def fleiss_kappa(label_matrix, n_classes):
    """label_matrix: [n_items, n_raters] of class indices."""
    m = np.asarray(label_matrix)
    n_items, n_raters = m.shape
    counts = np.stack([(m == k).sum(1) for k in range(n_classes)], 1)
    p_i = ((counts ** 2).sum(1) - n_raters) / (n_raters * (n_raters - 1))
    p_j = counts.sum(0) / (n_items * n_raters)
    pe = (p_j ** 2).sum()
    return float((p_i.mean() - pe) / (1 - pe)) if pe < 1 else 1.0


def evaluate(gold, preds, models, n_boot=1000, seed=0):
    """gold: {key: label}; preds: {key: preds-doc}; models: names. Returns the leaderboard dict."""
    keys = [k for k in gold if k in preds and all(m in preds[k]['clip'] for m in models)]
    y = np.array([CANONICAL.index(gold[k]) for k in keys])
    present = sorted(set(y.tolist()))
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(keys), len(keys)) for _ in range(n_boot)]

    def probs_for(name):
        if name == 'ensemble_mean':
            return np.mean([[preds[k]['clip'][m]['probs'] for k in keys] for m in models], 0)
        return np.array([preds[k]['clip'][name]['probs'] for k in keys])

    def frames_for(name, k):
        if name == 'ensemble_mean':
            per = [preds[k]['frames'][m] for m in models]
            return [None if f[0] is None else np.mean(f, 0) for f in zip(*per)]
        return preds[k]['frames'][name]

    rows = []
    for name in list(models) + (['ensemble_mean'] if len(models) > 1 else []):
        P = probs_for(name)
        pred = P.argmax(1)
        f1 = f1_score(y, pred, labels=present, average='macro', zero_division=0)
        f1_boot = [f1_score(y[b], pred[b], labels=present, average='macro', zero_division=0) for b in boots]
        neutral = CANONICAL.index('neutral')
        is_neutral = y == neutral
        # Fair score for models that cannot say "neutral" (e.g. C-MET's classifiers): only human-emotional clips
        emo = ~is_neutral
        emo_present = sorted(set(y[emo].tolist()))
        f1_emo = (f1_score(y[emo], pred[emo], labels=emo_present, average='macro', zero_division=0)
                  if emo.any() else None)
        metas = [preds[k].get('model_meta', {}).get(name) for k in keys] if name != 'ensemble_mean' else []
        classes = next((m['classes'] for m in metas if m), None) if metas else None
        rows.append({
            'model': name, 'n': len(keys), 'n_non_neutral': int(emo.sum()), 'classes': classes,
            'macro_f1_non_neutral': None if f1_emo is None else round(float(f1_emo), 4),
            'macro_f1': round(float(f1), 4),
            'macro_f1_ci95': [round(float(np.percentile(f1_boot, 2.5)), 4), round(float(np.percentile(f1_boot, 97.5)), 4)],
            'accuracy': round(float((pred == y).mean()), 4),
            'recall': {CANONICAL[c]: round(float((pred[y == c] == c).mean()), 4) for c in present},
            'neutral_fp_rate': round(float((pred[is_neutral] != neutral).mean()), 4) if is_neutral.any() else None,
            'ece': round(ece(P.max(1), pred == y), 4),
            'flip_rate': round(float(np.mean([flip_rate(frames_for(name, k)) for k in keys])), 4),
            'confusion': confusion_matrix(y, pred, labels=list(range(len(CANONICAL)))).tolist(),
        })
    rows.sort(key=lambda r: -r['macro_f1'])

    labels = {m: np.array([CANONICAL.index(preds[k]['clip'][m]['label']) for k in keys]) for m in models}
    pairwise = {f'{a}|{b}': round(float(cohen_kappa_score(labels[a], labels[b])), 4)
                for a, b in combinations(models, 2)}
    return {'n_clips': len(keys), 'labels': CANONICAL, 'gold_distribution':
            {CANONICAL[c]: int((y == c).sum()) for c in present}, 'leaderboard': rows,
            'agreement': {'cohen_kappa': pairwise,
                          'fleiss_kappa': round(fleiss_kappa(np.stack([labels[m] for m in models], 1),
                                                             len(CANONICAL)), 4) if len(models) > 1 else None}}
