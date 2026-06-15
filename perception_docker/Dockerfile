FROM wenbowen123/foundationpose:latest

ENV DEBIAN_FRONTEND=noninteractive

ENV POSE_ESTIMATION_DIR=/workspace/PoseEstimation
ENV FOUNDATIONPOSE_DIR=/workspace/PoseEstimation/submodules/FoundationPose
ENV SAM3_DIR=/workspace/PoseEstimation/submodules/SAM3
ENV SAM3_CONDA_DIR=/opt/sam3-conda

ENV EUROBIN_DIR=/workspace/eurobin

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    bash \
    ca-certificates \
    wget \
    curl \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# =========================
# PoseEstimation + submodules
# =========================

RUN git clone https://github.com/Giuseian/PoseEstimation.git ${POSE_ESTIMATION_DIR} && \
    cd ${POSE_ESTIMATION_DIR} && \
    git submodule update --init --recursive

# =========================
# eurobin / VLM stack
# =========================

RUN git clone https://github.com/Giuseian/eurobin.git ${EUROBIN_DIR} && \
    cd ${EUROBIN_DIR} && \
    git submodule update --init --recursive

# =========================
# FoundationPose
# =========================

WORKDIR ${FOUNDATIONPOSE_DIR}

RUN bash build_all.sh

RUN mkdir -p weights demo_data debug

# =========================
# Conda base: usata per SAM3 e VLM
# =========================

WORKDIR /workspace

RUN wget -O /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p ${SAM3_CONDA_DIR} \
    && rm /tmp/miniforge.sh

ENV PATH=${SAM3_CONDA_DIR}/bin:$PATH

# =========================
# SAM3 in ambiente separato
# =========================

RUN conda create -y -n sam3 python=3.12 \
    && conda clean -afy

WORKDIR ${SAM3_DIR}

RUN conda run -n sam3 python -m pip install --upgrade pip \
    && conda run -n sam3 pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && conda run -n sam3 pip install -e . \
    && conda run -n sam3 pip install \
        "setuptools<81" \
        einops \
        pycocotools \
        psutil \
        huggingface_hub

# =========================
# VLM / eurobin in ambiente separato
# =========================

RUN conda create -y -n vlm python=3.11 \
    && conda clean -afy

WORKDIR ${EUROBIN_DIR}

RUN conda run -n vlm python -m pip install --upgrade pip \
    && conda run -n vlm pip install -r ${EUROBIN_DIR}/requirements.txt

# =========================
# Comandi helper
# =========================

RUN echo '#!/usr/bin/env bash\nsource '${SAM3_CONDA_DIR}'/etc/profile.d/conda.sh\nconda activate sam3\ncd '${SAM3_DIR}'\nexec "$@"' > /usr/local/bin/run_sam3 \
    && chmod +x /usr/local/bin/run_sam3

RUN echo '#!/usr/bin/env bash\ncd '${FOUNDATIONPOSE_DIR}'\nexec "$@"' > /usr/local/bin/run_foundationpose \
    && chmod +x /usr/local/bin/run_foundationpose

RUN echo '#!/usr/bin/env bash\nsource '${SAM3_CONDA_DIR}'/etc/profile.d/conda.sh\nconda activate vlm\ncd '${EUROBIN_DIR}'\nexec "$@"' > /usr/local/bin/run_vlm \
    && chmod +x /usr/local/bin/run_vlm

WORKDIR ${POSE_ESTIMATION_DIR}

CMD ["/bin/bash"]