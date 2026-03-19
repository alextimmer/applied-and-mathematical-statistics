FROM docker.io/manimcommunity/manim:v0.18.1

USER root

# Install additional system dependencies for PyMC / CmdStanPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER manimuser

# Copy environment and install Python deps
COPY --chown=manimuser:manimuser pyproject.toml /manim/
COPY --chown=manimuser:manimuser src/ /manim/src/
COPY --chown=manimuser:manimuser environment.yml /manim/

RUN pip install --no-cache-dir \
    pymc>=5.10 \
    arviz>=0.17 \
    cmdstanpy>=1.2 \
    dowhy>=0.11 \
    statsmodels>=0.14 \
    seaborn>=0.13 \
    ipywidgets>=8.1 \
    nbval>=0.11 \
    && pip install --no-cache-dir -e /manim/

# Install CmdStan
RUN python -m cmdstanpy.install_cmdstan --cores 2

# Copy course content
COPY --chown=manimuser:manimuser . /manim

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
