FROM mambaorg/micromamba:1.5.8-jammy

# Use micromamba env activation in RUN/CMD
ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /app

# Copy project
COPY . /app

# Create environment with geospatial stack from conda-forge and pip deps
RUN micromamba create -y -n app -f /app/environment.yml \
    && micromamba clean --all --yes

# Add env to PATH
ENV PATH=/opt/conda/envs/app/bin:$PATH

# Streamlit defaults
ENV PORT=8501
EXPOSE 8501

CMD ["bash", "-lc", "streamlit run frontend/app.py --server.address 0.0.0.0 --server.port ${PORT}"]


