FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Mount videos at /in and results at /out; with no arguments every video in /in is processed.
ENV SUPERFOLEY_IN=/in SUPERFOLEY_OUT=/out
ENTRYPOINT ["superfoley"]
