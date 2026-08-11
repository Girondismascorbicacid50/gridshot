FROM ghcr.io/astral-sh/uv@sha256:2d890623d310b57771ce840f0da5eed5fc6d657da05ffaa45d82797b53fa3abc AS uv
FROM python@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

# opencv-python-headless needs libglib at import time on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/opt/gridshot
ENV PATH="/opt/gridshot/bin:${PATH}"

WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./
COPY gridshot ./gridshot
RUN uv sync --frozen --extra server --no-dev --no-cache

# The virtual environment stays outside /app, so the runtime source bind mount
# can replace /app without hiding the locked environment.
