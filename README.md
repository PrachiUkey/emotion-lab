# emotion-lab

Classify facial emotion **per frame** with several models, compare them against human labels, and browse
every clip with face boxes + per-model confidence in a password-protected dashboard. Also contains a working
[NED](https://github.com/foivospar/NED) pipeline for *editing* the emotion in a video.

```
emotion_lab/     python package: R2 + Lance data access, model plug-ins, per-frame inference, evaluation
scripts/         run_models.py, evaluate.py, upload/fetch checkpoints
configs/         models.yaml — the models under comparison
gold/            human labels (clip level)
web/             Next.js dashboard (deploy on Vercel, root directory = web)
ned/             NED emotion editing (MIT, Paraperas Papantoniou et al.) + torch-2 fixes + main.py
```

## Setup

```bash
cp .env.example .env            # fill in R2 keys — .env is gitignored, never commit it
git config core.hooksPath .githooks   # pre-commit hook blocks env files, keys and >5 MB files
pip install -r requirements.txt       # (python 3.10/3.11, CUDA GPU recommended)
```

Data comes from the R2 Lance datasets `fullbodydataset/{talkvid/lance,dataset_clip}/frames.lance`
(one row per frame: JPEG + 68 landmarks; no audio, fps assumed 25). Face boxes come from those landmarks.

## Run models and evaluate

```bash
# per-frame predictions + mp4 + thumbnail for every clip in a manifest (dataset, clip_id, fragment)
python scripts/run_models.py --manifest gold/labels_v0.csv
# score models against the human labels -> eval/leaderboard.json
python scripts/evaluate.py --gold gold/labels_v0.csv
```

Output goes to `$EMOTION_LAB_STORE`: `local:./data_store` (default) or `r2:dump-bucket/emotion-lab`.

### Adding a model

1. If it's a Hugging Face image classifier with emotion labels, just add it to `configs/models.yaml`:
   ```yaml
   my_model: {kind: hf_image, repo: org/name}
   ```
   Otherwise subclass `emotion_lab.models.base.EmotionModel` (set `labels`, implement `_predict(crops)`)
   and register a new `kind` in `emotion_lab/models/registry.py`. Unknown label names fail loudly;
   map them in `emotion_lab/labels.py`.
2. `python scripts/run_models.py --manifest ... --models my_model` (only the new model is computed).
3. `python scripts/evaluate.py --gold gold/labels_v0.csv`.

### How models are compared

Primary metric is **macro-F1** vs human labels (the data is mostly neutral, so plain accuracy rewards
guessing neutral), with a bootstrap 95% CI. Also: per-class recall, **neutral false-positive rate**
(talking mouths get misread as anger/disgust), calibration (ECE), per-frame flip rate, and inter-model
agreement (Cohen / Fleiss κ). A probability-average ensemble is scored too.

## Dashboard (`web/`)

```bash
cd web && cp .env.example .env.local   # gitignored
npm install && npm run dev
```

Deploy: import the GitHub repo in Vercel, set **Root Directory = `web`**, and add env vars
`DASHBOARD_PASSWORD`, `EMOTION_LAB_STORE=r2:dump-bucket/emotion-lab`, `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY` (use a read-only token). Media is served via 15-minute presigned R2 URLs.

## NED (emotion editing)

See `ned/` — `python ned/main.py --video in.mp4 --emotion happy --out out.mp4`. Env: `ned/setup_env.sh`.
Checkpoints: `ned/download_checkpoints.sh` (public sources) or `r2:dump-bucket/emotions_ckpt/`.
Some weights (FLAME, BFM albedo, FSGAN segmenter) are licence-gated by their authors; they are kept in a
private bucket only.

## Push to your GitHub

```bash
git remote add origin git@github.com:<you>/emotion-lab.git   # create the empty repo on GitHub first
git push -u origin main
```
(`upstream` points at the original NED repo.)
