# Credit_Card_Default_Prediction_Group13
# Chester Wang, HanChen Wang, Qurat-ul-Ain Azim, Renee Kwon
# Dec 8 2022

# Download base ubuntu image
FROM ubuntu:latest

# Set the installation process to be non-interactive
ARG DEBIAN_FRONTEND=noninteractive

# Install softwares (wget, GNU make, Python and R)
RUN apt-get update
RUN apt-get install -y wget software-properties-common
RUN apt-get install -y build-essential
RUN apt-get install -y python3 r-base
RUN apt-get install make -y

# These dependencies are essential for installing tidyverse and kableExtra
RUN apt-get install -y libxml2-dev libcurl4-openssl-dev libssl-dev libfontconfig1-dev

# Install Conda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path
ENV PATH=$CONDA_DIR/bin:$PATH


# Install packages via conda
RUN conda install --quiet -y -c defaults \
    "docopt=0.6.2" \
    "numpy=1.23.3" \
    "ipykernel" \
    "scikit-learn=1.1.2" \
    "pandas=1.4.4" \
    "requests>=2.24.0" \
    "scikit-learn=1.1.2" \
    "ipython>=7.15" \
    "selenium<4.3.0" \
    "matplotlib>=3.5.3" \
    "pandas-profiling" \
    "pandoc"

# Install packages via pip
RUN python3 -m pip install \
    "altair==4.2.0" \
    "altair_saver" \
    "scipy==1.9.2" \
    "joblib==1.1.0" \
    "psutil>=5.7.2" \
    "openpyxl>=3.0.0" \
    "xlrd>=2.0.1" \
    "xlwt>=1.3.0" \
    "vl-convert-python==0.5.0"

# Install R packages
RUN Rscript -e "install.packages('tidyverse')" && \
    Rscript -e "install.packages('knitr')" && \
    Rscript -e "install.packages('kableExtra')"
