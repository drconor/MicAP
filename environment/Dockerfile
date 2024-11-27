# hash:sha256:53e939fc7aaa5de7117a10a9e5e453660b15a97b09b2a0475032341d66ec5ada
FROM registry.codeocean.com/codeocean/miniconda3:4.8.2-python3.8-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0=2.56.4-0ubuntu0.18.04.6 \
        libsm6=2:1.2.2-1 \
        libxext6=2:1.3.3-1 \
        libxrender-dev=1:0.9.10-1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    opencv-python==4.3.0.36 \
    pandas==1.0.5 \
    pillow==6.2.2 \
    scikit-image==0.17.2 \
    scikit-learn==0.23.1 \
    xgboost==1.1.1 \
    -e git+https://github.com/whitews/cv2-extras.git@db304d838154b434fc019920a9ef08c04e588934#egg=cv2-extras
