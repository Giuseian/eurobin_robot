FROM wenbowen123/foundationpose:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV FOUNDATIONPOSE_DIR=/workspace/FoundationPose
ENV SAM3_DIR=/workspace/sam3
ENV SAM3_CONDA_DIR=/opt/sam3-conda

WORKDIR /workspace

# Pacchetti utili
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    bash \
    ca-certificates \
    wget \
    curl \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# =========================
# FoundationPose
# =========================

RUN git clone https://github.com/NVlabs/FoundationPose.git ${FOUNDATIONPOSE_DIR}

WORKDIR ${FOUNDATIONPOSE_DIR}

RUN bash build_all.sh

RUN mkdir -p weights demo_data debug

# =========================
# SAM3 in ambiente separato
# =========================

WORKDIR /workspace

# Installo Miniforge in una directory separata, così non tocco l'ambiente Python di FoundationPose
RUN wget -O /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p ${SAM3_CONDA_DIR} \
    && rm /tmp/miniforge.sh

ENV PATH=${SAM3_CONDA_DIR}/bin:$PATH

# Creo ambiente dedicato per SAM3
RUN conda create -y -n sam3 python=3.12 \
    && conda clean -afy

# Clono SAM3
RUN git clone https://github.com/facebookresearch/sam3.git ${SAM3_DIR}

WORKDIR ${SAM3_DIR}

# Installa PyTorch + SAM3.
# Nota: uso il comando indicato nel README SAM3 che hai incollato.
RUN conda run -n sam3 python -m pip install --upgrade pip \
    && conda run -n sam3 pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && conda run -n sam3 pip install -e . \
    && conda run -n sam3 pip install huggingface_hub

# Script comodo per attivare SAM3
RUN echo '#!/usr/bin/env bash\nsource '${SAM3_CONDA_DIR}'/etc/profile.d/conda.sh\nconda activate sam3\ncd '${SAM3_DIR}'\nexec "$@"' > /usr/local/bin/run_sam3 \
    && chmod +x /usr/local/bin/run_sam3

# Script comodo per FoundationPose
RUN echo '#!/usr/bin/env bash\ncd '${FOUNDATIONPOSE_DIR}'\nexec "$@"' > /usr/local/bin/run_foundationpose \
    && chmod +x /usr/local/bin/run_foundationpose

WORKDIR /workspace

CMD ["/bin/bash"]


# =========================
# VLM in ambiente separato
# =========================

ENV VLM_DIR=/workspace/vlm

WORKDIR /workspace

RUN conda create -y -n vlm python=3.11 \
    && conda clean -afy

RUN mkdir -p ${VLM_DIR}

# Dipendenze VLM equivalenti, senza path locali file://
RUN conda run -n vlm python -m pip install --upgrade pip \
    && conda run -n vlm pip install \
        annotated-types==0.7.0 \
        anyio==4.12.1 \
        certifi==2026.2.25 \
        cffi==2.0.0 \
        charset-normalizer==3.4.7 \
        cryptography==48.0.0 \
        distro==1.9.0 \
        google-auth==2.53.0 \
        google-genai==2.6.0 \
        h11==0.16.0 \
        httpcore==1.0.9 \
        httpx==0.28.1 \
        idna==3.11 \
        jiter==0.13.0 \
        matplotlib==3.10.8 \
        numpy==2.4.2 \
        openai==2.26.0 \
        packaging \
        pandas \
        pillow \
        pyasn1==0.6.3 \
        pyasn1_modules==0.4.2 \
        pycparser==3.0 \
        pydantic==2.12.5 \
        pydantic_core==2.41.5 \
        pyparsing \
        python-dateutil \
        python-dotenv==1.2.2 \
        requests==2.34.2 \
        six \
        sniffio==1.3.1 \
        tenacity==9.1.4 \
        tornado \
        tqdm==4.67.3 \
        typing-inspection==0.4.2 \
        typing_extensions==4.15.0 \
        urllib3==2.7.0 \
        websockets==16.0
