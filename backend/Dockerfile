FROM python:3.11-slim

WORKDIR /app

# Render's free tier has 512 MB. glibc otherwise keeps up to 8 malloc arenas per core for the
# request threads, and freed numpy buffers stay in them; 2 arenas keep the resident size down.
# Plain HTTP downloads from Hugging Face stream to disk; the Xet client sizes its buffers from
# the host's memory, not the container's limit.
ENV MALLOC_ARENA_MAX=2 \
    HF_HUB_DISABLE_XET=1

RUN useradd -m -u 1000 appuser && \
    mkdir -p /data/outputs /app/data && \
    chown -R appuser:appuser /data /app

# Exact versions from requirements.lock, so a new release of a dependency can't break a deploy.
COPY pyproject.toml requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock && \
    pip install --no-cache-dir --no-deps .

COPY app/ ./app/
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
