FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-web.txt /app/requirements-web.txt
RUN pip install --no-cache-dir -r /app/requirements-web.txt

COPY src/ /app/src/
COPY configs/ /app/config/

ENV PYTHONPATH=/app/src
ENV TOOL_FOR_LOGO_RUNTIME_DEFAULTS=/app/config/runtime.defaults.json
EXPOSE 8080
ENTRYPOINT ["python", "-m", "tool_for_logo", "web", "--host", "0.0.0.0", "--port", "8080"]
