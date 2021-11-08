FROM nvidia/cuda:11.0-base-ubuntu20.04

# non-interactive package installs by setting these up front
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ=UTC
ARG DEBIAN_FRONTEND=noninteractive

ARG CPU_OR_GPU=gpu

RUN apt-get update --fix-missing && \
    apt-get install --no-install-recommends -y \
        build-essential \
        git \
        libglu1-mesa \
        pkg-config \
        software-properties-common \
        unzip \
        wget && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/*

# Limited user
RUN groupadd -g 999 appuser && \
    useradd -r -u 1000 -g appuser appuser

RUN mkdir /codeexecution /envs /home/appuser && \
    chown -R appuser /codeexecution /envs /home/appuser /opt

USER 1000

# Install Miniconda as system Python
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ~/miniconda

ENV PATH ~/miniconda/bin:$PATH

# Create Python environment
COPY ./py-${CPU_OR_GPU}.yml /envs/
RUN ~/miniconda/bin/conda update conda -y && \
    ~/miniconda/bin/conda env create -f /envs/py-${CPU_OR_GPU}.yml && \
    ~/miniconda/bin/conda clean -a -y

# Create R environment
COPY ./r-${CPU_OR_GPU}.yml ./package-installs-${CPU_OR_GPU}.R /envs/
RUN ~/miniconda/bin/conda update conda -y && \
    ~/miniconda/bin/conda env create -f /envs/r-${CPU_OR_GPU}.yml && \
    ~/miniconda/bin/conda clean -a -y && \
    ~/miniconda/bin/conda clean -f -y && \
    ~/miniconda/envs/r-${CPU_OR_GPU}/bin/R -f /envs/package-installs-${CPU_OR_GPU}.R

COPY --chown=appuser:appuser entrypoint.sh /codeexecution/entrypoint.sh
COPY --chown=appuser:appuser scripts/metric.py scripts/score.py /codeexecution/scripts/
COPY --chown=appuser:appuser tests /codeexecution/tests/

# Execute the entrypoint.sh script inside the container when we do docker run
CMD ["/bin/bash", "/codeexecution/entrypoint.sh"]
