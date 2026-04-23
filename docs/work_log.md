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
