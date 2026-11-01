FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY config ./config
COPY data ./data
RUN pip install --no-cache-dir -e . --no-deps
# Build the index at image build time (falls back to the offline embedder if no network)
RUN sungrid ingest
EXPOSE 8000
CMD ["sungrid", "serve", "--host", "0.0.0.0", "--port", "8000"]
