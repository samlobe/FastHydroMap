FROM mambaorg/micromamba:1.5.8

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ARG TORCH_VARIANT=cpu

WORKDIR /opt/FastHydroMap

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
COPY --chown=$MAMBA_USER:$MAMBA_USER scripts/install_torch_pyg.sh /tmp/install_torch_pyg.sh

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes
RUN chmod +x /tmp/install_torch_pyg.sh && \
    /tmp/install_torch_pyg.sh "${TORCH_VARIANT}"

COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/FastHydroMap

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["micromamba", "run", "-n", "base", "python", "-m", "FastHydroMap"]
CMD ["--help"]
