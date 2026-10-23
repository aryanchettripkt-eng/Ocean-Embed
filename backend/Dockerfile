FROM python:3.11-slim

WORKDIR /app

RUN useradd -m -u 1000 appuser && \
    mkdir -p /data/outputs /app/data && \
    chown -R appuser:appuser /data /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY app/ ./app/
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
