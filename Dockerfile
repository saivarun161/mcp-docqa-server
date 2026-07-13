# Containerized docqa server on the Streamable HTTP transport.
#
#   docker build -t mcp-docqa-server .
#   docker run --rm -p 8000:8000 -v docqa-data:/data mcp-docqa-server
#
# Index a corpus into the same volume first, e.g.:
#   docker run --rm -v docqa-data:/data --entrypoint docqa-ingest \
#     mcp-docqa-server index --sample

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[pg,openai]"

RUN useradd --create-home docqa && mkdir /data && chown docqa:docqa /data
USER docqa

ENV DOCQA_SQLITE_PATH=/data/index.db

EXPOSE 8000
ENTRYPOINT ["docqa-server", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
