FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
COPY README.md .

RUN uv sync --no-dev

COPY src/ src/
COPY markets/ markets/
COPY presets/ presets/

RUN mkdir -p data logs config

EXPOSE 8787

ENTRYPOINT ["uv", "run"]
CMD ["learntrade", "serve", "--host", "0.0.0.0", "--port", "8787"]
