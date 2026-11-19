# syntax=docker/dockerfile:1

# This is our first build stage, it will not persist in the final image
FROM ubuntu AS intermediate
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client && rm -rf /var/lib/apt/lists/*
RUN mkdir -p -m 0700 /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts
# Download the computer vision framework
RUN --mount=type=ssh git clone git@github.com:pcr-upm/bmvc26_recognition.git bmvc26_recognition
ADD data /bmvc26_recognition/data
ADD splits /bmvc26_recognition/splits

# Copy the repository from the previous image
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
USER 0
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential wget openssh-client && rm -rf /var/lib/apt/lists/*
RUN mkdir -p -m 0700 /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN mkdir -p /home/username
WORKDIR /home/username
COPY --from=intermediate /bmvc26_recognition /home/username/bmvc26_recognition
LABEL maintainer="roberto.valle@upm.es"
# Setup conda environment
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/username/miniconda.sh
RUN chmod +x /home/username/miniconda.sh
RUN /home/username/miniconda.sh -b -p /home/username/conda
RUN /home/username/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /home/username/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN /home/username/conda/bin/conda create -n bmvc26 python=3.10
# Activate conda environment
ENV PATH=/home/username/conda/envs/bmvc26/bin:/home/username/conda/bin:${PATH}
# Make RUN commands use the new environment (source activate bmvc26)
SHELL ["conda", "run", "-n", "bmvc26", "/bin/bash", "-c"]
# Install dependencies
RUN conda run -n bmvc26 pip install pcr-framework pandas tqdm scikit-learn omegaconf einops torch torchvision torchinfo pytorch-lightning datasets taming-transformers-rom1504 clip kornia "tensorflow==2.15.1" "keras>=2.15,<2.16" "numpy==1.26.4" "gin-config==0.4.0"