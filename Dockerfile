# The app image: the chat UI and the API in one process.
#
#   docker compose up -d --build          # with the database (see docker-compose.yml)
#   docker build -t kith-rag-app .        # the image alone
#
# Stage 1 builds the UI with Node; only its output (frontend/dist) is copied into
# stage 2, a slim Python image that runs the API and serves that UI at /.
# No secrets are built in: OPENAI_API_KEY, DATABASE_URL and the Langfuse keys are
# passed as environment variables when the container starts.

# --- Stage 1: the chat UI ----------------------------------------------------------
FROM node:22-slim AS ui
WORKDIR /ui

# Dependencies first, so a source change doesn't reinstall them.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
# Optional, not secret: links the debug panel to Langfuse traces.
ARG VITE_LANGFUSE_PROJECT_URL=""
ENV VITE_LANGFUSE_PROJECT_URL=${VITE_LANGFUSE_PROJECT_URL}
# A failing test stops the build: a broken UI never becomes an image.
RUN npm test && npm run build


# --- Stage 2: the API ----------------------------------------------------------------
FROM python:3.13-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN useradd --create-home --uid 10001 app

# Exact versions from requirements.lock (scripts/lock_requirements.sh), so every build
# installs what was tested. pip check fails the build if the lock is incomplete.
COPY requirements.lock ./
RUN pip install --no-deps -r requirements.lock && pip check

COPY app/ app/
COPY ingestion/ ingestion/
COPY --from=ui /ui/dist frontend/dist

USER app
EXPOSE 8000

# Ready = the database answers and the catalog vocabulary loads.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4)" || exit 1

# One worker per container: the rate limiter and caches live in process memory.
# Scale by running more containers. --proxy-headers: behind Azure's ingress, the
# rate limit keys on the client's IP, not the proxy's.
CMD ["python", "-m", "uvicorn", "app.api.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips=*", "--no-access-log"]