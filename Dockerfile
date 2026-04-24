FROM mambaorg/micromamba:1.5.8

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ARG TORCH_VARIANT=cpu

WORKDIR /opt/FastHydroMap

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.dev.yml /tmp/environment.dev.yml

RUN micromamba install -y -n base -f /tmp/environment.dev.yml && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/FastHydroMap

RUN pip install --no-cache-dir -e . && \
    micromamba run -n base fasthydromap install-torch --variant "${TORCH_VARIANT}"

ENTRYPOINT ["micromamba", "run", "-n", "base", "python", "-m", "FastHydroMap"]
CMD ["--help"]
