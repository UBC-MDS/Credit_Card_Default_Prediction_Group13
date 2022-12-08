# Download base ubuntu image
FROM ubuntu:latest

# Set the installation process to be non-interactive
ARG DEBIAN_FRONTEND=noninteractive

# Install softwares (Python and R)
RUN apt-get update
RUN apt-get install -y wget software-properties-common
RUN apt-get install -y build-essential
RUN apt-get install -y python3 r-base

# Install Conda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash 
~/miniconda.sh -b -p /opt/conda

# Put Conda in path
ENV PATH=$CONDA_DIR/bin:$PATH

