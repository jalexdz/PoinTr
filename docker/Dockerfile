FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9" \
    PYTHONBUFFERED=1


RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git build-essential nano ca-certificates curl pkg-config tmux pcl-tools mesa-utils libgl1-mesa-glx libgl1-mesa-dri \
    wget cmake ninja-build evince \
    gcc-10 g++-10 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace
COPY . /workspace

RUN pip install -r requirements.txt

RUN bash install.sh

# PointNet++
RUN pip install --no-build-isolation \
    "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
RUN pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

WORKDIR /workspace/extensions/gridding
RUN python setup.py install

WORKDIR /workspace/extensions/chamfer_dist
RUN python setup.py install

WORKDIR /workspace/extensions/cubic_feature_sampling
RUN python setup.py install

WORKDIR /workspace/extensions/emd
RUN python setup.py install

WORKDIR /workspace/extensions/gridding_loss
RUN python setup.py install

WORKDIR /workspace
RUN python - <<'PY'
import torch
print("torch:", torch.__version__)
import pointnet2_ops
print("pointnet2_ops OK")
PY

ENV PYTHONPATH=/workspace DATA_ROOT=/data

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appuser && useradd -m -u ${USER_ID} -g ${GROUP_ID} appuser
USER appuser 

CMD ["/bin/bash"]
