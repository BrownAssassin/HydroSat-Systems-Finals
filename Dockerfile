ARG BASE_IMAGE=competition-base:pytorch2.5.1-cuda12.1-cudnn9
FROM ${BASE_IMAGE}

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src \
    INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    HYDROSAT_MODELS_DIR=/workspace/models

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
COPY models /workspace/models
COPY run.sh /workspace/run.sh

RUN chmod +x /workspace/run.sh

CMD ["/workspace/run.sh"]
