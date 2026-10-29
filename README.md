# Emotion Lab

**See what emotion every face shows, frame by frame — and find out which model gets it right.**

Emotion Lab runs several facial-expression models on talking-head video clips, draws a face box with the
predicted emotion and confidence on **every frame**, and scores each model against human labels so you can
pick the best one. It also includes a working **NED** pipeline that can *change* the emotion in a video
while keeping the speech.

| | |
|---|---|
| 🎞️ **Per-frame view** | Video player with face box + `happy 0.87` overlay, per model or all at once |
| 📊 **Model comparison** | Leaderboard against human labels: macro-F1, confusion matrices, model agreement |
| 🔍 **Find problems fast** | Filter clips where models disagree, or where a model differs from the human label |
| 🧩 **Plug-in models** | Add a model with one line of config; only the new model is computed |
| 🎭 **Emotion editing** | NED: turn a video happy / sad / angry / … with lip-sync preserved |
| 🔒 **Private by default** | Password-protected dashboard, private R2 bucket, secrets never committed |

---

## Contents

- [Quick start](#quick-start)
- [Credentials](#credentials)
- [Look at the results (dashboard)](#look-at-the-results-dashboard)
- [Run models on clips](#run-models-on-clips)
- [Compare models](#compare-models)
- [Add a new model](#add-a-new-model)
- [Change the emotion in a video (NED)](#change-the-emotion-in-a-video-ned)
- [Deploy the dashboard on Vercel](#deploy-the-dashboard-on-vercel)
- [Current results](#current-results)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Credits and licences](#credits-and-licences)

---

## Quick start

You need a Linux machine with an NVIDIA GPU (driver with CUDA ≥ 12.1) and ~25 GB of free disk.

```bash
git clone https://github.com/PrachiUkey/emotion-lab.git
cd emotion-lab
# 1. add your R2 keys (see "Credentials" below)
# 2. install everything
./setup.sh
```

`setup.sh` does the whole setup and is safe to run again — finished steps are skipped:

1. checks the GPU, driver and disk space
2. builds the **NED** Python env (torch + pytorch3d compiled for your GPU)
3. builds the **emotion-lab** Python env (the classifiers)
4. downloads all model checkpoints and verifies their checksums
5. installs the dashboard and turns on the git hook that blocks secrets

When it finishes you'll see a green checklist and the commands to use next.
To check an existing install at any time: `./setup.sh --check`.

<details>
<summary>Options</summary>

| Flag | What it does |
|---|---|
| `--check` | Only verify, change nothing |
| `--skip-web` | Don't install the dashboard |
| `--skip-ned` | Don't build the NED editing env |
| `--ckpt-source public` | Download checkpoints from public sources instead of R2 |
| `ENV_ROOT=/path ./setup.sh` | Put the Python envs somewhere else (default `/venv`, or `~/envs`) |

</details>

---

## Credentials

The clips live in Cloudflare R2. Create a file called `.env` in the repo root:

```ini
[r2]
type = s3
provider = Cloudflare
endpoint = https://<account_id>.r2.cloudflarestorage.com
access_key_id = <your-access-key-id>
secret_access_key = <your-secret-access-key>
region = auto
no_check_bucket = true
```

Or set `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` as environment variables instead.

> 🔒 `.env` is gitignored and a pre-commit hook refuses to commit env files or anything that looks like a key.
> Never paste keys into code, issues or chat.

| You want to… | Key needs |
|---|---|
| read clips, run models, download checkpoints | **read** access to `fullbodydataset` and `dump-bucket` |
| upload checkpoints or publish dashboard data | **write** access to `dump-bucket` |

---

## Look at the results (dashboard)

The dashboard shows every processed clip. Open one to play it with the face box, emotion and confidence on
each frame.

**On your own machine:**

```bash
cd web
cat > .env.local <<'EOF'
DASHBOARD_PASSWORD=pick-a-password
EMOTION_LAB_STORE=local:/absolute/path/to/emotion-lab/data_store
EOF
npm run build && npm start          # → http://localhost:3000
```

**On a remote GPU box** (e.g. Vast.ai) the dashboard runs as a background service on `127.0.0.1:3311`.
Reach it from your laptop with an SSH tunnel — nothing is exposed to the internet:

```bash
ssh -p <ssh-port> -L 3311:127.0.0.1:3311 root@<box-ip>
# then open http://localhost:3311
```

Manage the service with `supervisorctl status|restart emotion-lab-web`.

### What you'll see

- **Clips** — a grid of all clips with the human label and every model's label + confidence.
  Filters: dataset, model, emotion, *models disagree*, *model ≠ human label*.
- **Clip page** — the video with a face box and `emotion confidence` drawn on every frame.
  Switch between models (or show all), see the current frame's probability for each emotion,
  and a timeline strip per model — click it to jump to that moment.
- **Leaderboard** — how each model scores against the human labels.

---

## Run models on clips

Give it a list of clips (CSV with `dataset,clip_id,fragment` — e.g. `gold/labels_v0.csv`):

```bash
/venv/emotion-lab/bin/python scripts/run_models.py --manifest gold/labels_v0.csv
```

For each clip this saves per-frame predictions, an mp4 and a thumbnail to `data_store/`, and updates the
clip index. Refresh the dashboard to see them — no restart needed.

- Run just one model: `--models vit_fer`
- Try it on a few clips first: `--limit 5`
- Write straight to R2 instead of local disk: `--store r2:dump-bucket/emotion-lab`

> The source datasets store frames and face landmarks but **no audio and no frame rate**; videos are rebuilt
> at an assumed 25 fps, and labels are based on the face only.

---

## Compare models

```bash
/venv/emotion-lab/bin/python scripts/evaluate.py --gold gold/labels_v0.csv
```

Prints a leaderboard and saves it for the dashboard. Add `--min_confidence medium` to ignore uncertain
human labels.

| Metric | Why it matters |
|---|---|
| **Macro-F1** (main score) | Every emotion counts equally. Most clips are neutral, so plain accuracy rewards guessing "neutral". |
| **95% CI** | With few labels, scores are noisy — overlapping ranges mean "too close to call". |
| **Neutral false positives** | How often a calm talking face is called angry/disgusted/etc. — the most common failure. |
| **ECE** | Whether a model's confidence can be trusted (0 = perfectly calibrated). |
| **Frame flip rate** | How jumpy the per-frame predictions are. |
| **Cohen / Fleiss κ** | How much the models agree with each other. |

An **ensemble** (average of all models) is scored too.

---

## Add a new model

**Hugging Face image classifier** with emotion labels? Just add it to `configs/models.yaml`:

```yaml
my_model:
  kind: hf_image
  repo: some-org/some-emotion-model
```

**Anything else:** subclass `EmotionModel` in `emotion_lab/models/`, set `labels` and implement
`_predict(crops)` (RGB face crops in → probabilities out), then register its `kind` in
`emotion_lab/models/registry.py`. Unknown label names fail loudly — map them in `emotion_lab/labels.py`.

Then run only the new model and re-score:

```bash
/venv/emotion-lab/bin/python scripts/run_models.py --manifest gold/labels_v0.csv --models my_model
/venv/emotion-lab/bin/python scripts/evaluate.py --gold gold/labels_v0.csv
```

All models are mapped onto one label set: `neutral, happy, sad, surprised, fear, disgusted, angry, contempt`.

Finally, add a one-paragraph plain-language description of the model to `web/lib/glossary.ts` — it's what
people see when they hover over the model's name in the dashboard. (All other hover explanations live there too.)

---

## Change the emotion in a video (NED)

```bash
cd ned
/venv/NED/bin/python main.py --video in.mp4 --emotion happy --out out_happy.mp4
/venv/NED/bin/python main.py --video in.mp4 --emotion happy sad --out out.mp4   # 1st half happy, 2nd half sad
```

Emotions: `neutral happy sad surprised fear disgusted angry`. Lip movements follow the original speech.

- **New person:** the first run fine-tunes a face renderer on that person's video (**~1.5 hours** for a
  3-minute video on an RTX 4090, incl. preprocessing), then caches it — later runs for the same `--person` take minutes. More footage (a few minutes,
  face visible throughout) gives a sharper result.
- **Pretrained people:** `--renderer_ckpt ./checkpoints_tarantino` (also foxx, mcconaughey, mcdormand,
  pacino, roberts) skips training.
- **The face must be visible in every frame** — cut intros, title cards and cutaways first.

---

## Deploy the dashboard on Vercel

1. **Publish the data to R2** (needs a write-enabled key):
   ```bash
   /venv/emotion-lab/bin/python scripts/sync_store.py --dry-run   # preview
   /venv/emotion-lab/bin/python scripts/sync_store.py
   ```
2. **Import the repo** at [vercel.com/new](https://vercel.com/new) and set **Root Directory = `web`**.
3. **Add environment variables:**

   | Name | Value |
   |---|---|
   | `DASHBOARD_PASSWORD` | the password your team will use |
   | `EMOTION_LAB_STORE` | `r2:dump-bucket/emotion-lab` |
   | `R2_ENDPOINT` | `https://<account_id>.r2.cloudflarestorage.com` |
   | `R2_ACCESS_KEY_ID` | a **read-only** key |
   | `R2_SECRET_ACCESS_KEY` | its secret |

4. **Deploy.** Every `git push` redeploys. After changing an env var, redeploy for it to take effect.

Videos are served with 15-minute signed links, so the bucket stays private. New data published with
`sync_store.py` appears without redeploying.

---

## Current results

53 human-labelled clips (labels by visual inspection, clip level). On the 42 medium/high-confidence labels:

| Model | Trained on | Macro-F1 (95% CI) | Calls neutral faces emotional |
|---|---|---|---|
| `vit_fer` | FER2013 | 0.67 (0.49–0.81) | 43% |
| ensemble (mean) | — | 0.66 (0.50–0.80) | 43% |
| `hsemotion_b0` | AffectNet / VGAF | 0.55 (0.39–0.69) | 75% |
| `hsemotion_b2` | AffectNet | 0.40 (0.21–0.54) | 86% |

**No clear winner yet.** The confidence ranges overlap, and the labels are almost all neutral or happy
(sad 4, surprised 1, disgusted 1). Every model confuses talking mouths with emotions. Next steps:
~500 labels covering every emotion, and models that are less fooled by speech.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Dashboard says **"No index.json in the store yet"** | No data published yet — run `run_models.py` (local) or `sync_store.py` (Vercel). |
| **Wrong password / can't log in** | It's `DASHBOARD_PASSWORD` (`web/.env.local` locally, Vercel → Settings → Environment Variables). After changing it, restart the service or redeploy. |
| **`AccessDenied`** uploading to R2 | Your key is read-only — create one with *Object Read & Write* on `dump-bucket`. |
| `No R2 credentials` | Create `.env` — see [Credentials](#credentials). |
| `setup.sh`: Blackwell / compute capability ≥ 10 | Not supported by the pinned torch 2.4.1 + CUDA 12.1 build. |
| `No space left on device` | NED writes large frame folders under `ned/runs/` — delete old runs there. |
| NED crashes in `eye_landmarks` | The video starts without a visible face — trim the start. |
| Pre-commit hook blocks your commit | It found an env file, a key-like value or a file > 5 MB. Remove it from the commit (weights belong in R2). |

---

## Project layout

```
emotion-lab/
├── setup.sh                 one-command setup / --check
├── emotion_lab/             Python package
│   ├── data.py              read clips from the R2 Lance datasets, face boxes from landmarks
│   ├── models/              model plug-ins (HSEmotion, Hugging Face, …) + registry
│   ├── infer.py             per-frame predictions, mp4 + thumbnail
│   ├── evaluate.py          metrics
│   └── store.py, r2.py      local / R2 storage, credentials
├── scripts/
│   ├── run_models.py        run models on a list of clips
│   ├── evaluate.py          leaderboard vs human labels
│   ├── sync_store.py        publish dashboard data to R2
│   ├── fetch_checkpoints.py restore checkpoints from R2 (checksum-verified)
│   └── upload_checkpoints.py back up checkpoints to R2
├── configs/
│   ├── models.yaml          models being compared
│   └── checkpoints_manifest.json
├── gold/labels_v0.csv       human labels
├── web/                     Next.js dashboard
└── ned/                     NED emotion editing (+ main.py)
```

Data layout (same locally in `data_store/` and in R2 under `dump-bucket/emotion-lab/`):

```
index.json                  clip list + per-model clip labels
clips/<dataset>/<clip>.mp4  video
thumbs/<dataset>/<clip>.jpg thumbnail
preds/<dataset>/<clip>.json per-frame boxes + per-model probabilities
gold/labels.json            human labels
eval/leaderboard.json       model comparison
```

Checkpoints are backed up in `r2:dump-bucket/emotions_ckpt/`.

---

## Credits and licences

- **NED** — Paraperas Papantoniou et al., *Neural Emotion Director*, CVPR 2022 ([code](https://github.com/foivospar/NED), MIT). Adapted here for torch 2 / modern GPUs.
- **DECA**, **FLAME**, **Basel Face Model** and the **FSGAN** face segmenter are licence-gated by their
  authors. Their weights are kept in a private bucket only — check their licences before any other use.
- **HSEmotion** — A. Savchenko ([code](https://github.com/HSE-asavchenko/face-emotion-recognition), Apache-2.0).
- **vit-face-expression** — [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression) on Hugging Face.
