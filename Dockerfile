# Eval runner image. The model itself runs in a separate Ollama container (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so code edits don't bust the dependency layer cache.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY config ./config
COPY data ./data
COPY prompts ./prompts
COPY baseline ./baseline

# Run as a non-root user.
RUN useradd --create-home evaluser && mkdir -p /app/reports && chown -R evaluser /app
USER evaluser

ENTRYPOINT ["python", "-m", "evalgate"]
CMD ["run", "--output-dir", "reports"]
