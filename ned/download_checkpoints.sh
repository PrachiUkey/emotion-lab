#!/bin/bash
# Fetch every checkpoint NED needs into this repo. Run from the repo root.
# Needs: gdown, huggingface_hub (pip install gdown huggingface_hub), unzip.
#
# FLAME, the BFM albedo and FSGAN's segmenter are normally gated (FLAME/BFM registration,
# FSGAN request form). They are pulled from unofficial public mirrors below; the original
# licenses still apply.
set -euo pipefail
cd "$(dirname "$0")"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

# --- NED (public Google Drive, from the README) ---
gdown --folder https://drive.google.com/drive/folders/1fLAsB2msBcLnRJWlixXt-hJ8FeX3Az6T -O "$STAGE/manip"
gdown --folder https://drive.google.com/drive/folders/1vBVeiBvVP_fZ5jPSv7yd7OsdiI22Mwnd -O "$STAGE/renderers"
unzip -q -o "$STAGE/manip/manipulator_checkpoints.zip" -d .
unzip -q -o "$STAGE/renderers/Meta-renderer/checkpoints_meta-renderer.zip" -d .
for a in foxx:Foxx/checkpoints_foxx mcconaughey:McConaughey/McConaughey_checkpoints \
         mcdormand:McDormand/checkpoints_mcdormand pacino:Pacino/checkpoints_pacino \
         roberts:Roberts/checkpoints_robert tarantino:Tarantino/checkpoints_tarantino; do
    mkdir -p "checkpoints_${a%%:*}"
    unzip -q -o "$STAGE/renderers/${a#*:}.zip" -d "checkpoints_${a%%:*}"
done

# --- DECA model (public Google Drive, from the DECA README) ---
gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O DECA/data/deca_model.tar

# --- Gated files via public mirrors ---
python - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil
shutil.copy(hf_hub_download("icampara/smirkapp5", "assets/FLAME2020/generic_model.pkl", repo_type="space"),
            "DECA/data/generic_model.pkl")
shutil.copy(hf_hub_download("fachrinnk4869/deca_dataset", "FLAME_albedo_from_BFM.npz", repo_type="dataset"),
            "DECA/data/FLAME_albedo_from_BFM.npz")
EOF
# FSGAN segmenter: inside the SCMCL checkpoint bundle (github.com/ZH-Xu410/SCMCL), ~8.7 GB zip
gdown 1CTjWl--IDxa2CyogtfLVTY518C8cex_u -O "$STAGE/scmcl_ckpts.zip"
unzip -q -o -j "$STAGE/scmcl_ckpts.zip" scmcl_ckpts/segmentation/lfw_figaro_unet_256_2_0_segmentation_v1.pth \
      -d preprocessing/segmentation/

echo "All checkpoints in place."
