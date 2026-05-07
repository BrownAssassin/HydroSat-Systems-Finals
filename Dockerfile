ARG BASE_IMAGE=10.200.99.202:15080/zero2x002/competition-base:pytorch2.5.1-cuda12.1-cudnn9
FROM ${BASE_IMAGE}

WORKDIR /workspace

ARG PIP_CACHE_DIR=/tmp/pip-cache

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src \
    INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    MODEL_DIR=/workspace/models \
    HYDROSAT_MODELS_DIR=/workspace/models

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --cache-dir "${PIP_CACHE_DIR}" -r /workspace/requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple

COPY . /workspace

RUN chmod +x /workspace/run.sh

CMD ["/workspace/run.sh"]
