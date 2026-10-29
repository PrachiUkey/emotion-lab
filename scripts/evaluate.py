"""Score every model in the store against gold labels; writes eval/leaderboard.json.

    python scripts/evaluate.py --gold gold/labels_v0.csv [--min_confidence medium]
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotion_lab.evaluate import evaluate  # noqa: E402
from emotion_lab.store import get_json, open_store, put_json  # noqa: E402

CONF_ORDER = {'low': 0, 'medium': 1, 'high': 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--store', default=None)
    ap.add_argument('--models', default=None, help='comma-separated; default = every model present in all preds')
    ap.add_argument('--min_confidence', default='low', choices=list(CONF_ORDER))
    args = ap.parse_args()

    store = open_store(args.store)
    gold_rows = [r for r in csv.DictReader(open(args.gold))
                 if CONF_ORDER[r.get('label_confidence', 'high')] >= CONF_ORDER[args.min_confidence]]
    gold = {f"{r['dataset']}/{r['clip_id']}": r['emotion'] for r in gold_rows}
    preds, missing = {}, []
    for k in gold:
        if store.exists(f'preds/{k}.json'):
            preds[k] = get_json(store, f'preds/{k}.json')
        else:
            missing.append(k)
    if missing:
        print(f'!!! {len(missing)} gold clips have no predictions (run scripts/run_models.py): {missing[:5]}...')
    models = args.models.split(',') if args.models else sorted(set.intersection(*(set(p['clip']) for p in preds.values())))

    res = evaluate(gold, preds, models)
    res['gold_file'], res['min_confidence'] = args.gold, args.min_confidence
    put_json(store, 'eval/leaderboard.json', res)
    put_json(store, 'gold/labels.json', {f"{r['dataset']}/{r['clip_id']}": {
        'emotion': r['emotion'], 'confidence': r.get('label_confidence'), 'annotator': r.get('annotator'),
        'notes': r.get('notes', '')} for r in csv.DictReader(open(args.gold))})

    print(f"gold clips: {res['n_clips']}  (min_confidence={args.min_confidence})  {res['gold_distribution']}")
    print(f"{'model':16s} {'macroF1':>8s} {'95% CI':>15s} {'acc':>6s} {'neutralFP':>9s} {'ECE':>6s} {'flip':>6s}")
    for r in res['leaderboard']:
        ci = f"[{r['macro_f1_ci95'][0]:.2f},{r['macro_f1_ci95'][1]:.2f}]"
        print(f"{r['model']:16s} {r['macro_f1']:8.3f} {ci:>15s} {r['accuracy']:6.2f} {r['neutral_fp_rate']:9.2f} "
              f"{r['ece']:6.3f} {r['flip_rate']:6.3f}  recall={r['recall']}")
    print('agreement:', res['agreement'])


if __name__ == '__main__':
    main()
