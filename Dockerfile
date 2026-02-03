FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS builder
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE.md /app/
COPY src /app/src
RUN pip install --upgrade pip \
 && pip wheel --wheel-dir /wheels .

FROM base
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels rcsb-embedding-model \
 && rm -rf /wheels
ENTRYPOINT ["inference"]
