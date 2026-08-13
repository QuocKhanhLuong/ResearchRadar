FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system researchradar \
    && adduser --system --ingroup researchradar --home /app researchradar

# Copy only packaging inputs and application source. In particular, no local
# .env file, database, model weights, or development artifact is added to image.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && mkdir -p /app/data \
    && chown -R researchradar:researchradar /app

USER researchradar

VOLUME ["/app/data"]

CMD ["python", "-m", "research_radar.main"]
