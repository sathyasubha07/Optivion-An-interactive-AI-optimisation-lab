# OptiVision — An Interactive AI Optimisation Lab

**See inside the black box.** OptiVision is a web lab where each module poses a real-world task as an
optimisation problem, runs the module's own solver on your input (or a labelled sample), and shows the
whole path: the problem, the maths, a replay of the recorded iterations, the results with optimality
(KKT) diagnostics, a plain-language explanation, and downloadable outputs.

MFC course project — Team 08.

## Modules

| Module | Method |
|---|---|
| 1 — Network Intrusion Detection | Soft-margin linear SVM + L1 sparsity, convex QP solved with OSQP |
| 4 — Satellite Super-Resolution | Weighted least squares + L1 sparsity, solved with ISTA |

This is a group project. Each module lives in its own folder, and the integration layer never edits module
source: it calls each module's entry point through an adapter in `backend/adapters/`.

## Quick start (Windows PowerShell)

Requires Python 3.10+ (tested on 3.14) and Node.js 20+ (tested on 24). Run the two halves in two terminals,
both from the repository root.

**Backend** (http://127.0.0.1:8000):

```powershell
pip install -r requirements.txt; uvicorn backend.main:app --reload
```

**Frontend** (http://localhost:5173):

```powershell
cd frontend; npm install; npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api` and `/outputs` to the backend on port 8000.

Optional: to keep packages out of your global Python, create a virtual environment first:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

### Data

- **Module 4** offers UC Merced Land Use samples when `Module4_Satellite_SuperResolution/data/UCMerced_LandUse/`
  exists locally (the folder is gitignored, ~318 MB). Without it, the synthetic sample still works. To download
  the dataset, run Module 4's own `ensure_uc_merced_dataset()` once.
- **Module 1** has no bundled real dataset. The sample uses the module's synthetic flows, and is labelled as
  such. To test real traffic, upload a CIC-IDS2017 MachineLearningCSV file (it must have a `Label` column).
  Large files are capped at 20,000 rows and a 3,000-row stratified sample. With hyperparameter tuning, a
  78-feature upload takes about 30 s.

## How it fits together

```
frontend/  React + Vite + Tailwind + Framer Motion + Recharts + KaTeX
  └─ renders everything from GET /modules and the job result. It contains no module-specific code.
backend/
  main.py        FastAPI app: module registry, uploads, background jobs, static outputs
  registry.py    one entry per module: text, maths, upload rules, and its adapter
  adapters/      one file per module; calls the module's own functions
    base.py      the shared adapter contract and result shape
  runtime/       per-job inputs and outputs (gitignored)
Module1_Network_Intrusion_Detection/   (owned by its author; not modified)
Module4_Satellite_SuperResolution/     (owned by its author; not modified)
```

### Adding a new module

1. Write `backend/adapters/<name>.py` exposing `ADAPTER = Adapter(run=..., list_samples=..., make_sample=...)`.
   `run(ctx)` calls the module and returns the result shape documented in `backend/adapters/base.py`
   (summary metrics, metric groups, replay series, images, plots, charts, tables, explanation, notes, downloads).
2. Add one entry to `MODULES` in `backend/registry.py` (copy an existing one), with `"_adapter": <name>.ADAPTER`.
3. Add the module's dependencies to `requirements.txt`.

No frontend change is needed: the module grid, the 7-step workspace and the results views are all generic.

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/modules` | All registry entries, including the available samples |
| GET | `/modules/{id}` | One module |
| POST | `/modules/{id}/run` | multipart form: `file` **or** `sample_id`. Returns `{job_id}` |
| GET | `/jobs/{job_id}` | `queued` / `running` / `succeeded` / `failed`, plus `result` or `message` |
| GET | `/contributors` | Credits table (built from the registry) |
| GET | `/outputs/...` | Files written by runs (images, plots, CSV, TXT, JSON) |

Errors: unknown module → 404; coming-soon module, missing or duplicate input, unknown sample, or empty file → 400;
wrong file type → 415; file too large → 413. If the module itself fails, the job ends as `failed` with the
module's own error message.

## Honesty rules

- The UI shows only numbers returned by the backend for that run. Nothing is hard-coded, and a missing
  value is shown as "not computed".
- Iteration views are labelled **replays of recorded iterations**. Module 1's OSQP solver does not record
  per-iteration values, so its replay step says so instead of drawing a curve.
- Samples are labelled. Synthetic data is never presented as a benchmark result.

## License

See `LICENSE`.
