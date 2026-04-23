# Decision Intelligence Assistant

Full-stack support-ticket assistant for the AIE Bootcamp Week 3 project. The app
compares four outputs for one customer-support ticket:

- RAG answer using retrieved historical tickets
- non-RAG answer without retrieved context
- trained ML priority prediction
- LLM zero-shot priority prediction

The goal is to reason about quality, latency, and cost instead of treating the
LLM as the only possible solution.

## Architecture

```text
React frontend
  -> FastAPI /api/v1/analyze
      -> retrieval service: Chroma persistent vector store when available
      -> retrieval fallback: persisted TF-IDF similarity index
      -> Groq LLM adapter when GROQ_API_KEY is configured
      -> local deterministic fallback when no LLM key is configured
      -> sklearn priority model artifacts when exported
      -> weak-label rule fallback when artifacts are missing
      -> JSONL request logging
```

Chroma is implemented as the first retrieval backend because it satisfies the
vector-store requirement in local persistent mode. The backend also has a TF-IDF
fallback because Chroma can fail in restricted local environments; that fallback
keeps the demo usable while surfacing the source of retrieved results.

## Repository Layout

```text
backend/      FastAPI app, services, schemas, training script, Dockerfile
frontend/     React + Vite + TypeScript UI, Dockerfile
notebooks/    EDA, weak labeling, feature engineering, model comparison
data/         demo sample plus folders for raw/interim/processed data
artifacts/    generated model/vectorizer/retrieval artifacts
logs/         JSONL request logs
docs/         step-by-step work log
```

## Environment Variables

Copy `.env.example` to `.env` for local Docker runs and fill in real values when
needed.

Important variables:

- `GROQ_API_KEY`: optional. If missing, the app uses deterministic local fallback
  responses.
- `GROQ_MODEL`: Groq model name.
- `DATASET_PATH`: dataset used by retrieval.
- `CHROMA_PERSIST_DIRECTORY`: persistent Chroma storage path.
- `LOG_PATH`: JSONL query log path.
- `VITE_API_BASE_URL`: frontend API target.

## Run With Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

Open:

- Frontend: <http://localhost:5173>
- Backend docs: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/api/v1/health>

Stop the stack:

```bash
docker compose down
```

Remove named volumes if you want a clean retrieval/log state:

```bash
docker compose down -v
```

## Local Development

Backend dependencies are managed with `uv`.

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Training Artifacts

The committed `data/sample/customer_support_sample.csv` is a tiny smoke-test
dataset so the project runs in a fresh clone. For the real project run, download
the Kaggle dataset and use `twcs.csv`, not the 100-row `sample.csv`.

The notebook prefers `twcs.csv` when downloading through `kagglehub`. You can
also place it manually at `data/raw/twcs.csv`.

Export artifacts:

```bash
cd backend
uv run python scripts/train_priority_model.py ^
  --dataset ..\data\sample\customer_support_sample.csv ^
  --artifacts-dir ..\artifacts
```

For the full 10k-sample workflow, run
`notebooks/decision_intelligence_assistant.ipynb`, which:

- cleans customer tweets
- builds a representative sample
- documents the weak-label priority rule
- compares TF-IDF, engineered, and combined features
- evaluates Logistic Regression, LinearSVC, and Random Forest
- exports backend artifacts

## Logging

Every `/api/v1/analyze` request writes one JSON line to `logs/requests.jsonl`
or the configured `LOG_PATH`. The event includes:

- request ID and input ticket
- retrieved sources and similarity scores
- RAG and non-RAG answers
- ML and LLM priority predictions
- subsystem latency and LLM cost fields

## Design Decisions

- Weak labels are transparent rules, not human truth. Reported ML metrics measure
  how well the model reproduces those weak labels.
- ML prediction is recommended for high-volume routing because it has zero
  per-call LLM cost and low latency.
- LLM zero-shot classification is useful for low-confidence, disputed, or
  high-risk tickets where reasoning text helps reviewers.
- Local fallback behavior is deliberate so the app is reviewable without
  committing secrets.

## Known Limitations

- The demo sample is intentionally small and not suitable for final metrics.
- Real metrics should come from the notebook run on a representative 10k sample
  from `twcs.csv`.
- In the local sandbox used during development, Chroma initialization produced a
  disk I/O error, so the backend fell back to the persisted TF-IDF index. Docker
  or an unrestricted local run should use Chroma normally.
