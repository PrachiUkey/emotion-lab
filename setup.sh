#!/bin/bash
# One-shot setup for emotion-lab on a fresh Linux + NVIDIA GPU machine. Safe to re-run: finished steps are skipped.
#
#   ./setup.sh                      # everything
#   ./setup.sh --skip-web           # no dashboard deps
#   ./setup.sh --ckpt-source public # skip R2, fetch checkpoints from public sources/mirrors
#   ./setup.sh --check              # only verify an existing setup
#
# Creates two Python envs under $ENV_ROOT (default /venv if writable, else ~/envs):
#   NED          python 3.10, torch 2.4.1+cu121, pytorch3d (built for this GPU) -> ned/ editing pipeline
#   emotion-lab  classifiers/eval; inherits NED's torch via --system-site-packages (saves ~5 GB)
# Needs: git, curl, bzip2, unzip, an NVIDIA driver supporting CUDA >= 12.1, ~25 GB free disk.
# R2 checkpoint download needs .env (copy .env.example) with keys that can read dump-bucket.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

SKIP_NED=0; SKIP_LAB=0; SKIP_WEB=0; SKIP_CKPT=0; CHECK_ONLY=0; CKPT_SOURCE=auto
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-ned) SKIP_NED=1 ;; --skip-lab) SKIP_LAB=1 ;; --skip-web) SKIP_WEB=1 ;; --skip-ckpt) SKIP_CKPT=1 ;;
        --ckpt-source) CKPT_SOURCE="$2"; shift ;;
        --check) CHECK_ONLY=1 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m!!! SETUP FAILED: %s\033[0m\n' "$*" >&2; exit 1; }

if [ -z "${ENV_ROOT:-}" ]; then
    if [ -d /venv ] && [ -w /venv ]; then ENV_ROOT=/venv; else ENV_ROOT="$HOME/envs"; fi
fi
NED_ENV="$ENV_ROOT/NED"
LAB_ENV="$ENV_ROOT/emotion-lab"
mkdir -p "$ENV_ROOT"

# ---------------------------------------------------------------- prerequisites
log "Checking prerequisites (env root: $ENV_ROOT)"
for c in git curl unzip bzip2; do command -v "$c" >/dev/null || die "'$c' is not installed"; done
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: an NVIDIA GPU + driver is required"
CC_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
# Header says "CUDA Version: X" on older drivers, "CUDA UMD Version: X" on newer ones
DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA[A-Z ]*Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)
echo "GPU compute capability $CC_CAP, driver supports CUDA ${DRV_CUDA:-unknown}"
awk -v c="$CC_CAP" 'BEGIN{exit !(c >= 10.0)}' && \
    die "compute capability $CC_CAP (Blackwell) needs CUDA >= 12.8 wheels; this setup pins torch 2.4.1+cu121"
if [ -z "$DRV_CUDA" ]; then
    warn "could not read the driver's CUDA version from nvidia-smi; continuing (torch's CUDA check will catch problems)"
elif awk -v d="$DRV_CUDA" 'BEGIN{exit !(d < 12.1)}'; then
    die "driver supports only CUDA $DRV_CUDA; need >= 12.1"
fi
FREE_GB=$(df -Pk "$REPO" | awk 'NR==2{print int($4/1048576)}')
[ "$FREE_GB" -lt 8 ] && warn "only ${FREE_GB} GB free on $(df -P "$REPO" | awk 'NR==2{print $6}'); a full setup needs ~25 GB"

find_conda() {
    for c in "$(command -v mamba || true)" "$(command -v conda || true)" /opt/miniforge3/bin/mamba \
             "$HOME/miniforge3/bin/mamba" /opt/conda/bin/conda; do
        [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return; }
    done
}

ned_ok() { [ -x "$NED_ENV/bin/python" ] && "$NED_ENV/bin/python" - <<'EOF' >/dev/null 2>&1
import torch, pytorch3d, chumpy, face_alignment, facenet_pytorch, skimage, cv2
from pytorch3d.renderer.mesh import rasterize_meshes
assert torch.cuda.is_available()
EOF
}
lab_ok() { [ -x "$LAB_ENV/bin/python" ] && "$LAB_ENV/bin/python" -c \
    "import torch, transformers, onnxruntime, lance, boto3, hsemotion_onnx, yaml, sklearn" >/dev/null 2>&1; }

# ---------------------------------------------------------------- NED env
build_ned_env() {
    local CONDA; CONDA=$(find_conda)
    if [ -z "$CONDA" ]; then
        log "Installing Miniforge to $HOME/miniforge3"
        curl -fsSL -o /tmp/miniforge.sh \
            "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
        bash /tmp/miniforge.sh -b -p "$HOME/miniforge3" && rm /tmp/miniforge.sh
        CONDA="$HOME/miniforge3/bin/mamba"
    fi
    log "Creating NED env at $NED_ENV with $CONDA"
    [ -x "$NED_ENV/bin/python" ] || "$CONDA" create -y -p "$NED_ENV" python=3.10
    local P="$NED_ENV/bin"
    "$P/pip" install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
    # CUDA 12.1 toolchain matching torch, to compile pytorch3d
    "$CONDA" install -y -p "$NED_ENV" -c nvidia/label/cuda-12.1.1 -c conda-forge \
        cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-libraries-dev=12.1 "gxx_linux-64<12" ffmpeg
    "$P/pip" install fvcore iopath
    log "Building pytorch3d v0.7.7 for sm_${CC_CAP/./} (takes ~10 min)"
    PATH="$P:$PATH" CUDA_HOME="$NED_ENV" TORCH_CUDA_ARCH_LIST="$CC_CAP" FORCE_CUDA=1 MAX_JOBS="$(nproc)" \
    CC="$P/x86_64-conda-linux-gnu-gcc" CXX="$P/x86_64-conda-linux-gnu-g++" \
    CPATH="$NED_ENV/targets/x86_64-linux/include" LIBRARY_PATH="$NED_ENV/targets/x86_64-linux/lib:$NED_ENV/lib" \
        "$P/pip" install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
    "$P/pip" install "numpy<2" scipy scikit-image opencv-python "face-alignment==1.3.5" munch yacs dominate \
        "moviepy==1.0.3" imageio imageio-ffmpeg loguru torchfile tqdm pyyaml pillow matplotlib pandas \
        scikit-learn tensorboard gdown huggingface_hub
    "$P/pip" install --no-deps facenet-pytorch==2.5.3
    "$P/pip" install --no-build-isolation --no-deps chumpy==0.70
    # chumpy imports numpy aliases removed in numpy >= 1.24
    local CH; CH=$("$P/python" -c "import importlib.util as u; print(u.find_spec('chumpy').origin)")
    sed -i 's/^from numpy import bool, int, float, complex, object, unicode, str, nan, inf$/from numpy import nan, inf\nbool, int, float, complex, object, unicode, str = bool, int, float, complex, object, str, str/' "$CH"
}

# ---------------------------------------------------------------- emotion-lab env
build_lab_env() {
    log "Creating emotion-lab env at $LAB_ENV (on top of NED's torch)"
    command -v uv >/dev/null || "$NED_ENV/bin/pip" install -q uv
    local UV; UV=$(command -v uv || echo "$NED_ENV/bin/uv")
    [ -x "$LAB_ENV/bin/python" ] || "$UV" venv -q "$LAB_ENV" --python "$NED_ENV/bin/python" --system-site-packages
    # torch comes from the NED env; install the rest of requirements.txt, keeping numpy < 2
    grep -vE '^(--extra-index-url|torch==)' requirements.txt > "$LAB_ENV/requirements.lab.txt"
    VIRTUAL_ENV="$LAB_ENV" "$UV" pip install -q -r "$LAB_ENV/requirements.lab.txt"
    VIRTUAL_ENV="$LAB_ENV" "$UV" pip install -q "numpy<2"
}

# ---------------------------------------------------------------- checkpoints
fetch_ckpts() {
    if [ "$CKPT_SOURCE" != public ]; then
        log "Fetching checkpoints from r2:dump-bucket/emotions_ckpt (sha256-verified)"
        if [ ! -f .env ] && [ -z "${R2_ACCESS_KEY_ID:-}" ]; then
            warn "no .env / R2_* env vars: cannot use R2"
        elif "$LAB_ENV/bin/python" scripts/fetch_checkpoints.py; then
            return 0
        else
            warn "R2 fetch failed (see above)"
        fi
        [ "$CKPT_SOURCE" = r2 ] && die "checkpoints not restored from R2"
    fi
    log "Falling back to public sources + mirrors (ned/download_checkpoints.sh; downloads ~10 GB incl. an 8.7 GB bundle)"
    PATH="$NED_ENV/bin:$PATH" bash ned/download_checkpoints.sh || die "public checkpoint download failed"
    "$LAB_ENV/bin/python" -c "from hsemotion_onnx.facial_emotions import get_model_path as g; import urllib.request; \
        [g(m) for m in ('enet_b0_8_best_vgaf', 'enet_b2_8')]"
}

# ---------------------------------------------------------------- web
setup_web() {
    if ! command -v node >/dev/null && [ -s /opt/nvm/nvm.sh ]; then . /opt/nvm/nvm.sh >/dev/null; fi
    if ! command -v node >/dev/null && [ -s "$HOME/.nvm/nvm.sh" ]; then . "$HOME/.nvm/nvm.sh" >/dev/null; fi
    if ! command -v node >/dev/null; then
        log "Installing Node 20 via nvm"
        curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
        . "$HOME/.nvm/nvm.sh" && nvm install 20
    fi
    local MAJOR; MAJOR=$(node -p 'process.versions.node.split(".")[0]')
    [ "$MAJOR" -ge 18 ] || die "node $MAJOR is too old for Next.js 14 (need >= 18)"
    if [ web/node_modules/.package-lock.json -nt web/package-lock.json ]; then
        log "Dashboard deps up to date, skipping"
        return
    fi
    log "Installing dashboard deps (web/)"
    (cd web && npm ci --no-audit --no-fund --loglevel=error)
}

# ---------------------------------------------------------------- verification
verify() {
    log "Verifying"
    local fail=0
    if [ "$SKIP_NED" = 0 ]; then
        if ned_ok; then echo "ok   NED env ($NED_ENV): torch+CUDA, pytorch3d"; else warn "NED env broken or missing"; fail=1; fi
    fi
    if [ "$SKIP_LAB" = 0 ]; then
        if lab_ok; then echo "ok   emotion-lab env ($LAB_ENV)"; else warn "emotion-lab env broken or missing"; fail=1; fi
    fi
    if [ "$SKIP_CKPT" = 0 ]; then
        if "$LAB_ENV/bin/python" scripts/fetch_checkpoints.py --check >/dev/null 2>&1; then
            echo "ok   all checkpoints in configs/checkpoints_manifest.json present (sha256 verified)"
        else
            # public-source downloads have identical bytes for most files; list exactly what differs
            warn "checkpoint check failed:"; "$LAB_ENV/bin/python" scripts/fetch_checkpoints.py --check | grep -v '^ok' || true
            fail=1
        fi
    fi
    if [ "$SKIP_LAB" = 0 ]; then
        if "$LAB_ENV/bin/python" -c "from emotion_lab.models.registry import load_models; \
                print('ok   models load:', [m.name for m in load_models()])" 2>/dev/null; then :; else
            warn "classifier models failed to load"; fail=1; fi
    fi
    if [ "$SKIP_WEB" = 0 ]; then
        if [ -d web/node_modules/next ]; then echo "ok   web deps installed"; else warn "web/node_modules missing"; fail=1; fi
    fi
    [ -f .env ] && echo "ok   .env present (gitignored)" || warn ".env missing: cp .env.example .env and fill in R2 keys"
    [ "$(git config core.hooksPath || true)" = .githooks ] && echo "ok   pre-commit secret guard enabled" \
        || { warn "git hooks not enabled"; fail=1; }
    [ "$fail" = 0 ] || die "verification failed (see warnings above)"
    log "Setup complete"
    cat <<EOF
  NED editing:   $NED_ENV/bin/python ned/main.py --video in.mp4 --emotion happy --out out.mp4
  classifiers:   $LAB_ENV/bin/python scripts/run_models.py --manifest gold/labels_v0.csv
  evaluation:    $LAB_ENV/bin/python scripts/evaluate.py --gold gold/labels_v0.csv
  dashboard:     cd web && cp .env.example .env.local && npm run dev
EOF
}

# ---------------------------------------------------------------- main
git config core.hooksPath .githooks
[ -f .env ] || { [ -n "${R2_ACCESS_KEY_ID:-}" ] || warn ".env not found: copy .env.example to .env and add R2 keys (needed for R2 data/checkpoints)"; }

if [ "$CHECK_ONLY" = 1 ]; then verify; exit 0; fi

if [ "$SKIP_NED" = 0 ]; then
    if ned_ok; then log "NED env OK, skipping ($NED_ENV)"; else build_ned_env; ned_ok || die "NED env still broken after build"; fi
fi
if [ "$SKIP_LAB" = 0 ]; then
    [ -x "$NED_ENV/bin/python" ] || die "emotion-lab env needs the NED env (it reuses its torch); don't combine --skip-ned on a fresh machine"
    if lab_ok; then log "emotion-lab env OK, skipping ($LAB_ENV)"; else build_lab_env; lab_ok || die "emotion-lab env still broken after build"; fi
fi
if [ "$SKIP_CKPT" = 0 ]; then
    if "$LAB_ENV/bin/python" scripts/fetch_checkpoints.py --check >/dev/null 2>&1; then
        log "Checkpoints OK, skipping"
    else
        fetch_ckpts
    fi
fi
[ "$SKIP_WEB" = 0 ] && setup_web
verify
