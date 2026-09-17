# syntax=docker/dockerfile:1

# This is our first build stage, it will not persist in the final image
FROM ubuntu as intermediate
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client && rm -rf /var/lib/apt/lists/*
RUN mkdir -p -m 0700 /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts
# Download the computer vision framework
RUN --mount=type=ssh git clone git@github.com:pcr-upm/bmvc26_recognition.git bmvc26_recognition
ADD data /bmvc26_recognition/data

# Copy the repository from the previous image
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
USER 0
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
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
RUN /home/username/conda/bin/conda create -n bmvc26 python=3.11
# Activate conda environment
ENV PATH /home/username/conda/envs/bmvc26/bin:/home/username/conda/bin:$PATH
# Make RUN commands use the new environment (source activate bmvc26)
SHELL ["conda", "run", "-n", "bmvc26", "/bin/bash", "-c"]
# Install dependencies
RUN conda run -n bmvc26 pip install --no-cache-dir torchcodec==0.2.1 --index-url https://download.pytorch.org/whl/cu124
RUN conda run -n bmvc26 pip install --no-cache-dir nvidia-npp-cu12==12.4.1.87
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/npp/lib
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
