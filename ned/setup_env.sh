#!/bin/bash
# Build the "NED" conda env for modern GPUs (tested: RTX 4090, sm_89).
# Replaces the upstream environment.yml (torch 1.7.1 / CUDA 11.0), which predates Ada/Hopper.
set -euo pipefail
export PATH=/opt/miniforge3/bin:$PATH
mamba create -y -n NED python=3.10
P=$(conda env list | awk '$1=="NED"{print $NF}')

$P/bin/pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.1 toolchain to compile pytorch3d against torch's CUDA version
mamba install -y -n NED -c nvidia/label/cuda-12.1.1 -c conda-forge \
    cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-libraries-dev=12.1 "gxx_linux-64<12" ffmpeg

$P/bin/pip install fvcore iopath
PATH=$P/bin:$PATH CUDA_HOME=$P TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" FORCE_CUDA=1 \
CC=$P/bin/x86_64-conda-linux-gnu-gcc CXX=$P/bin/x86_64-conda-linux-gnu-g++ \
CPATH=$P/targets/x86_64-linux/include LIBRARY_PATH=$P/targets/x86_64-linux/lib:$P/lib \
    $P/bin/pip install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"

$P/bin/pip install "numpy<2" scipy scikit-image opencv-python "face-alignment==1.3.5" munch yacs dominate \
    "moviepy==1.0.3" imageio imageio-ffmpeg loguru torchfile tqdm pyyaml pillow matplotlib pandas \
    scikit-learn tensorboard gdown huggingface_hub
$P/bin/pip install --no-deps facenet-pytorch==2.5.3
$P/bin/pip install --no-build-isolation --no-deps chumpy==0.70
# chumpy imports aliases removed in numpy>=1.24
sed -i 's/^from numpy import bool, int, float, complex, object, unicode, str, nan, inf$/from numpy import nan, inf\nbool, int, float, complex, object, unicode, str = bool, int, float, complex, object, str, str/' \
    $P/lib/python3.10/site-packages/chumpy/__init__.py
echo "NED env ready at $P"
