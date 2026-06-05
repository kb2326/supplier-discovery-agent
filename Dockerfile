# Container for the supplier-discovery demo. Works on any host that runs Docker
# (Render, Railway, Fly.io, Hugging Face Spaces, Cloud Run, etc.).
FROM python:3.12-slim

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY app/ ./app/
COPY data/ ./data/
COPY static/ ./static/

# Hosts inject $PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

# Shell form so $PORT expands at runtime.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
