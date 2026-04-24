# Work Log

This log records the build steps for the Decision Intelligence Assistant repo.
It is written during development so the final submission can explain what was
done, which tools were used, what failed, and how each issue was handled.

## 2026-04-24

### 1. Initial Repo Audit

- Location: local workspace at
  `e:\Extra Courses\AIE Bootcamp\Week 3\Project 3\decision-intelligence-assistant`.
- Action: listed the existing repository folders and tracked git state.
- Result: found an existing scaffold with `backend/`, `frontend/`, `notebooks/`,
  `data/`, `artifacts/`, `logs/`, `.env.example`, `.gitignore`, and a git
  history.
- Git state: only `notebooks/decision_intelligence_assistant.ipynb` was modified
  before the new implementation work started.

### 2. Requirements Review

- Location: local project brief PDF at
  `e:\Extra Courses\AIE Bootcamp\Week 3\Project 3\Decision-Intelligence-Assistant (1).pdf`.
- Action: extracted the text with Python and `pypdf` because `pdftotext` was not
  available on the machine.
- Result: confirmed the required deliverables:
  - notebook with EDA, weak labeling, feature engineering, and model comparison
  - FastAPI backend
  - React frontend
  - vector store or justified persistent in-process store
  - Docker Compose
  - service Dockerfiles
  - `.env.example`
  - README with architecture, run steps, design decisions, and limitations
  - logging setup that captures queries, retrieval, outputs, latency, cost, and
    errors

### 3. Coding Guidelines Review

- Location: local coding guideline PDF at
  `e:\Extra Courses\AIE Bootcamp\Prerequisites\AIE_Bootcamp_Coding_Guidelines.pdf`.
- Action: extracted the text with Python and `pypdf`.
- Result: captured relevant rules for this project:
  - no committed secrets or virtual environments
  - use `.env.example` for dummy config
  - write clear README documentation
  - use typed Python functions and PEP 8 style
  - use structured logging instead of `print` for backend operations
  - validate API boundaries with Pydantic
  - use meaningful git commits

### 4. Dependency Tooling Check

- Action: checked package tooling.
- Result: `uv 0.11.7` is installed.
- Bug encountered: an early attempt used `pip install` before the instructor's
  `uv` preference was re-stated. The install failed because no package versions
  were reachable in the restricted environment, and the later escalation request
  was rejected.
- Fix: stop using `pip` for this project and use `uv` commands from this point
  forward.

### 5. First Documentation Commit

- Action: created this work log in `docs/work_log.md`.
- Git bug encountered: `git add docs/work_log.md` failed with
  `Unable to create '.git/index.lock': Permission denied`.
- Investigation: checked `.git` for an existing `index.lock`; no stale lock was
  present. The issue was write permission to the git metadata under the current
  sandbox.
- Fix: reran `git add` and `git commit` with approval for elevated git metadata
  access.
- Commit created: `6bdcb4e docs: Add project work log`.

### 6. Backend Dependency Setup With uv

- Action: used `uv add --project backend ...` to add backend dependencies.
- Result: `uv` created `backend/.venv` and updated backend dependency metadata.
- Bug encountered: the first non-elevated `uv add` failed because the sandbox
  could not access the uv cache at `C:\Users\Pc\AppData\Local\uv\cache`.
- Fix: reran `uv add` with approved access. The install completed.
- Warning encountered: uv fell back from hardlinking to full file copies. This is
  a performance warning only and does not affect correctness.

### 7. Training Script Smoke-Test Bugs

- Action: added `backend/scripts/train_priority_model.py` and ran it on the
  committed demo sample.
- Bug encountered: `classification_report` crashed because one test split did
  not include every weak-label class.
- Fix: passed explicit label indexes to `classification_report` so missing
  classes are reported with zero support.
- Bug encountered: `RandomForestClassifier(n_jobs=-1)` failed in the Windows
  sandbox with `PermissionError: [WinError 5] Access is denied` while creating
  multiprocessing/threading primitives.
- Fix: changed Random Forest to `n_jobs=1` for reliable local and review runs.

### 8. Dataset Clarification

- User clarification: `sample.csv` is only a 100-row dataset sample, while
  `twcs.csv` is the full dataset.
- Planned fix: prefer `twcs.csv` in notebook/download logic and use the small
  committed `data/sample/customer_support_sample.csv` only as a smoke-test
  fallback.

### 9. Backend Implementation

- Action: added FastAPI `/api/v1/analyze` route with schemas, service layer,
  retrieval, ML inference, LLM adapter, and JSONL query logging.
- Files added or changed:
  - `backend/app/routers/analyze.py`
  - `backend/app/schemas/analysis.py`
  - `backend/app/services/*`
  - `backend/app/main.py`
  - `backend/app/core/config.py`
- Design: Groq is used when `GROQ_API_KEY` is configured. Without a key, the
  backend returns deterministic local fallback answers and priority labels so
  the app remains reviewable without secrets.
- Bug encountered: Chroma persistent mode failed in the local sandbox with
  `disk I/O error`.
- Fix: kept Chroma as the first retrieval path, but added a persistent TF-IDF
  retrieval fallback so `/api/v1/analyze` does not crash in restricted
  environments.

### 10. Frontend Implementation

- Action: added a Vite React TypeScript frontend with ticket input, RAG vs
  non-RAG answer panels, ML vs LLM priority cards, source tickets, latency, and
  cost displays.
- Files added:
  - `frontend/package.json`
  - `frontend/index.html`
  - `frontend/src/App.tsx`
  - `frontend/src/lib/api.ts`
  - `frontend/src/types/analysis.ts`
  - `frontend/src/styles.css`
- Bug encountered: first `npm install` failed because the sandbox used
  `only-if-cached` mode and the packages were not cached.
- Fix: reran `npm install` with approved network/cache access.
- Bug encountered: first build failed because React/Vite type declarations were
  missing.
- Fix: installed `@types/react` and `@types/react-dom`, then added
  `frontend/src/vite-env.d.ts`.
- Bug encountered: Vite/esbuild failed with `spawn EPERM` in the sandbox.
- Fix: reran `npm run build` with approved process-spawn access.

### 11. Deployment Files and Verification

- Action: added Dockerfiles, `.dockerignore` files, `docker-compose.yml`, and
  root `README.md`.
- Backend Dockerfile uses the official uv Python image and `uv sync --frozen`.
- Frontend Dockerfile uses `npm ci`.
- Verification:
  - `docker compose config` passed.
  - Backend `TestClient` health check returned `{\"status\": \"ok\"}`.
  - Backend `POST /api/v1/analyze` returned HTTP 200 with ML prediction output.
  - `npm run build` passed for the frontend.
- Cleanup: `frontend/tsconfig.tsbuildinfo` was generated by the TypeScript
  build. It was unstaged, added to `.gitignore`, and removed from disk because
  build caches should not be committed.

### 12. Docker Build Verification Blocker

- Action: ran `docker compose build`.
- Result: Docker CLI exists, but the build failed before image creation.
- Bug/blocker encountered: Docker could not connect to
  `npipe:////./pipe/dockerDesktopLinuxEngine`.
- Interpretation: Docker Desktop's Linux engine is not running on the machine.
- Fix needed outside the repo: start Docker Desktop, wait for the engine to be
  ready, then rerun `docker compose build` or `docker compose up --build`.

### 13. GitHub Publication Check

- Action: checked git remotes and GitHub CLI.
- Result: no git remote is configured and `gh` is not installed.
- Next step: create an empty GitHub repository in the browser, then add it as
  `origin` and push `main`.

### 14. Docker Compose Success

- User action: Docker Desktop was opened and the engine became available.
- Action: ran `docker compose up --build -d`.
- Result: backend and frontend images built successfully and both containers
  started.
- Verification:
  - `GET http://localhost:8000/api/v1/health` returned `{\"status\":\"ok\"}`.
  - `POST http://localhost:8000/api/v1/analyze` returned HTTP 200.
  - Docker retrieval used `chroma-persistent-hashing-embeddings`, confirming the
    vector store path works inside Docker.
- Issue found: the backend container initially used `rule-fallback` for ML
  prediction because the named artifact volume starts empty.
- Fix: updated `backend/Dockerfile` so container startup trains/exports model
  artifacts from the mounted dataset before launching Uvicorn.

### 15. Notebook Output Execution

- Issue reported: notebook code cells did not show saved outputs.
- Investigation: only the first two code cells had execution counts and outputs;
  the remaining edited cells had not been executed after implementation.
- Action: added notebook execution dev dependencies with `uv`:
  `nbformat`, `nbclient`, and `ipykernel`.
- Action: registered the backend uv environment as a Jupyter kernel named
  `decision-intelligence-backend`.
- Bug encountered: registering/executing the kernel first failed with Windows
  permission errors while writing Jupyter runtime/kernel files.
- Fix: reran with approved filesystem access and set `JUPYTER_RUNTIME_DIR`
  inside the project during execution.
- Bug encountered: notebook execution failed on the 100-row raw `sample.csv`
  because one weak-label class had only one example, so sklearn could not use a
  stratified split.
- Fix: updated the notebook and training script to use stratified splitting only
  when every class has at least two examples; otherwise they use a regular
  reproducible split and print a warning.
- Result: notebook execution completed successfully and every code cell now has
  saved outputs.

### 16. Full Dataset 10k Sample Run

- User action: downloaded and unzipped the full Kaggle dataset, then placed
  `twcs.csv` in `data/raw`.
- Action: reran the notebook with `twcs.csv` present.
- Result: `data/sample/customer_support_sample.csv` was regenerated as a
  10,000-row sample from the full dataset.
- Issue found: the original brand inference produced 10,000 unique `brand_hint`
  values, which meant it was often selecting numeric customer IDs instead of
  support brand handles.
- Fix: updated notebook brand inference to inspect linked tweets and only treat
  non-numeric author IDs as likely brands.
- Result after fix: the 10k sample has 109 inferred brand values. Some rows
  remain `unknown` where the conversation links do not expose a brand.

### 17. Editor and TypeScript Warnings

- Issue found: frontend `tsconfig.json` used `moduleResolution: "Node"`, which
  VS Code warned is deprecated for future TypeScript versions.
- Fix: changed it to `moduleResolution: "Bundler"`, which matches Vite.
- Issue found: VS Code/Pylance showed missing-import warnings for backend
  dependencies such as `groq` and `chromadb`.
- Fix: added `pyrightconfig.json` pointing analysis at `backend/.venv` and
  adding `backend` to `extraPaths`.

### 18. Docker 10k Runtime Verification

- Issue found: backend startup became too slow when it attempted to run the full
  training-script model comparison on the 10k sample.
- Fix: added `--deploy-fast` mode to `backend/scripts/train_priority_model.py`.
  Docker startup now trains the deployable combined-feature `LinearSVC` artifact
  on the 10k sample, while the notebook remains the source for the broader model
  comparison.
- Verification:
  - frontend `npm run build` passed after the TypeScript config update.
  - Docker backend artifact metadata reports `dataset_rows = 10000`.
  - Docker backend artifact model reports `LinearSVC`.
  - `/api/v1/analyze` returned Chroma source metadata:
    `chroma-persistent-hashing-embeddings`.
  - A second post-index analyze request completed in about 1.5 seconds.
- Security note: `docker compose config` expands values from `.env`, including
  secrets. Avoid sharing that output, and rotate any API key that appears in
  terminal logs or screenshots.
