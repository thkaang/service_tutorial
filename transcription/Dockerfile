# Use an official Python runtime as the base image
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /transcription_pipe

# Use default option for installing apt packages
ENV DEBIAN_FRONTEND=noninteractive

# Install packages
RUN apt-get update || true
RUN apt-get install python3.10 -y
RUN apt-get install python3.10-venv -y
RUN apt-get install python3.10-dev -y
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN apt-get install libopenmpi-dev -y
RUN apt-get install curl -y
RUN apt-get install wget -y
RUN apt-get install automake -y
RUN apt-get install autoconf -y
RUN apt-get install libtool -y
RUN apt-get install pkg-config -y
RUN apt-get install vim -y
RUN apt-get install ffmpeg -y
RUN apt-get install build-essential -y
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
ENV NVM_DIR=/root/.nvm
RUN . "$NVM_DIR/nvm.sh" \
    && nvm install 24.11.1
ENV PATH="$NVM_DIR/versions/node/v24.11.1/bin/:$PATH"

RUN rm -rf /var/lib/apt/lists/*

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py
RUN python get-pip.py
RUN rm get-pip.py

# Install any needed packages specified in requirements.txt
COPY ./requirements.txt /transcription_pipe/.
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt --no-cache-dir

# Copy the current directory contents into the container
COPY ./audio_data /transcription_pipe/audio_data
COPY ./transcription_pipe.py /transcription_pipe/.
COPY ./transcription_pipe_test.py /transcription_pipe/.
COPY ./pre_setting.py /transcription_pipe/.
COPY ./server.py /transcription_pipe/.
COPY ./ui_server.py /transcription_pipe/.
COPY ./config.json /transcription_pipe/.
COPY ./core /transcription_pipe/core
COPY ./server /transcription_pipe/server

# npm build
WORKDIR /transcription_pipe/server
RUN npm install --force
RUN npm audit fix --force || echo "npm audit fix returned non-zero code, ignoring..."
RUN npm run build
WORKDIR /transcription_pipe
