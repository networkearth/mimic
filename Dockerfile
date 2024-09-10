FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install build-essential wget git vim curl

RUN apt-get -y install python3 python-is-python3
RUN apt-get -y install python3-pip

# Only way to successfully install h5py is through conda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
RUN bash Miniconda3-latest-Linux-aarch64.sh -b
RUN root/miniconda3/bin/conda init
RUN rm Miniconda3-latest-Linux-aarch64.sh

# Needed for Tensorflow to work
RUN root/miniconda3/bin/conda install -y h5py