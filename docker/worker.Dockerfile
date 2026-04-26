FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-worker.txt /app/requirements-worker.txt
RUN pip install --no-cache-dir -r /app/requirements-worker.txt

COPY src/ /app/src/
COPY configs/ /app/config/

ENV PYTHONPATH=/app/src
ENV TOOL_FOR_LOGO_RUNTIME_DEFAULTS=/app/config/runtime.defaults.json
ENTRYPOINT ["python", "-m", "tool_for_logo", "daemon", "--poll-interval", "5"]
