FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DAMSELFISH_CONFIG=/app/config.yml

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates git openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY damselfish ./damselfish
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 damselfish \
    && mkdir -p /app/data \
    && chown -R damselfish:damselfish /app

USER damselfish
EXPOSE 8086

CMD ["damselfish", "--config", "/app/config.yml", "--host", "0.0.0.0", "--port", "8086"]
