# Stage 1: build the SPA
FROM node@sha256:6c74791e557ce11fc957704f6d4fe134a7bc8d6f5ca4403205b2966bd488f6b3 AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Stage 2: python app image + built SPA
FROM ghcr.io/astral-sh/uv@sha256:2d890623d310b57771ce840f0da5eed5fc6d657da05ffaa45d82797b53fa3abc AS uv
FROM python@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
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
RUN uv sync --frozen --no-dev --extra server --no-cache
COPY --from=web /web/dist ./web/dist
CMD ["uvicorn", "gridshot.server.app:app", "--host", "0.0.0.0", "--port", "8800"]
