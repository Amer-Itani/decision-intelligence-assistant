FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY scripts ./scripts

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8000

CMD ["sh", "-c", "python scripts/train_priority_model.py --deploy-fast --dataset ${DATASET_PATH:-/app/data/sample/customer_support_sample.csv} --artifacts-dir /app/artifacts && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
