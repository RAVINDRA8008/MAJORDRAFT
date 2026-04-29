FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY deploy/backend/requirements.txt /app/deploy/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/backend/requirements.txt

COPY config /app/config
COPY src /app/src
COPY deploy/backend /app/deploy/backend
COPY deploy/frontend /app/deploy/frontend
COPY checkpoints /app/checkpoints

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --app-dir deploy/backend --host 0.0.0.0 --port ${PORT:-8000}"]
