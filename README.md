# Module 1 — Network Intrusion Detection

Part of the **OPTIVION** project (an interactive AI optimisation lab). This module is a **backend-only** system that classifies network traffic as **Benign** or **Attack**, using a soft-margin linear SVM with L1 sparsity, trained as a convex quadratic program (QP) and solved with OSQP.

This is an academic/research prototype, not a production intrusion-detection product. There is **no frontend, dashboard, or UI** in this module — it exposes a FastAPI JSON API and a command-line interface, intended to be consumed by a future OPTIVION frontend.

---

## 1. What this module does

- Classifies network flows as:
  - `Benign` → label `0`
  - `Attack` → label `1`
- Trains a **linear SVM with L1 sparsity**, formulated as a convex quadratic program
- Solves it using **OSQP** (an ADMM-based QP solver), not `sklearn.svm.SVC`
- Returns full optimisation diagnostics alongside predictions: model weights, sparsity, support vectors, solver status, objective value, and KKT (optimality) conditions
- Evaluates using **detection-relevant metrics** (attack recall, false-positive rate), not accuracy alone — a model that looks accurate but misses attacks is not treated as successful

---

## 2. Why these design choices

**Why SVM:** A linear SVM has a convex training problem with a well-defined margin, and its dual multipliers identify support vectors. That makes optimisation diagnostics (objective value, constraint satisfaction, KKT residuals) meaningful — which is the actual point of this module. It is not claimed to beat every modern IDS model; tree ensembles or deep models can outperform it on some splits, but that's outside this module's scope.

**Why L1 sparsity:** The L1 penalty pushes many feature weights toward exactly zero, so the module can report which traffic features the detector actually relies on. This is model-level feature selection, not a causal explanation of what causes attacks.

---

## 3. Mathematical formulation

The intended optimisation problem:

```
min_{w,b,ξ}   ½‖w‖² + C·Σᵢξᵢ + λ‖w‖₁

subject to:   yᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ,   ξᵢ ≥ 0
```

where training labels `yᵢ ∈ {−1, +1}` (Attack = +1, Benign = −1).

Since `‖w‖₁` is non-smooth, it is **not** passed to OSQP directly as a quadratic term. Instead, it is reformulated using a split-variable transformation:

```
w = w⁺ − w⁻,    w⁺ ≥ 0,   w⁻ ≥ 0,    ‖w‖₁ = 1ᵀ(w⁺ + w⁻)
```

This yields the convex QP actually solved:

```
min   ½(w⁺−w⁻)ᵀ(w⁺−w⁻) + C·Σᵢξᵢ + λ·1ᵀ(w⁺+w⁻)
```

subject to the original margin/slack constraints and non-negativity of `w⁺, w⁻, ξ`. The Hessian block on `(w⁺, w⁻)` is:

```
[ I   −I ]
[ −I   I ]
```

which is positive semi-definite, confirming convexity.

When class-balanced weighting is enabled, `C` is replaced per-sample with `Cᵢ = C · n / (2·n_yᵢ)` (the standard "balanced" weighting scheme). This affects training only.

---

## 4. Solver — OSQP

**OSQP** (Operator Splitting Quadratic Program solver, ADMM-based) is used because the split-variable formulation is a convex QP with sparse linear constraints, and OSQP returns a full primal solution, dual variables, residuals, iteration count, and solve status — rather than acting as a black box.

If OSQP reports a non-`solved` status, training fails explicitly with a clear error rather than silently returning a bad model.

---

## 5. KKT (optimality) diagnostics

Computed directly from the solved QP, not estimated:

- Primal feasibility of margin and non-negativity constraints
- Stationarity residual: `‖Pz + q + Aᵀy‖∞` (using OSQP's dual variables and the symmetric Hessian)
- Complementary slackness on bound constraints
- Dual box feasibility: `0 ≤ αᵢ ≤ Cᵢ` for margin multipliers
- Bias stationarity proxy: `|Σαᵢyᵢ|`

Note: ADMM-based solvers like OSQP can have moderate residuals even when the reported status is `solved` — this is expected numerical behaviour on real, large datasets, not a correctness failure. A feature weight `wⱼ` is treated as zero when `|wⱼ| ≤ sparsity_threshold` (default `1e-6`).

---

## 6. Dataset — CIC-IDS2017

Primary dataset: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Canadian Institute for Cybersecurity).

**Setup:**
1. Download the machine-learning CSV files (not the raw PCAP captures)
2. Place the CSV file(s) in:
   ```
   module1/data/cicids2017/
   ```
3. No hard-coded local paths are used — the directory is configurable via YAML.

Label mapping: original label `BENIGN` → `0`; any other original label → `1`. Original attack-category strings are preserved internally for future per-attack-type analysis; multi-class classification is not the goal of this version.

Unit tests use small **synthetic** (fake, generated) traffic and do not require the real dataset. Synthetic results are never reported as CIC-IDS2017 performance.

---

## 7. Preprocessing pipeline

1. Load one or more CIC-IDS2017 CSV files
2. Strip column name whitespace; drop configured identifier columns
3. Coerce all features to numeric (invalid values become missing)
4. Map infinite values to missing
5. Drop columns that are entirely missing
6. Drop duplicate rows **before** splitting (prevents the same row appearing in both train and test)
7. Optional stratified development subsample (for tractable debugging — documented in results, not treated as a full evaluation)
8. Stratified train/test split (default 80/20, fixed random seed)
9. Class-imbalance handling on the **training set only**
10. Missing-value imputation and feature scaling, **fit on the training set only**
11. Drop columns that are constant on the training set

The test set is never resampled and never used to fit any preprocessing step — preventing data leakage.

---

## 8. Class imbalance handling

CIC-IDS2017 is often skewed toward benign or attack traffic depending on which capture days are combined. Default strategy: **balanced class weighting inside the QP** (`Cᵢ = C · n / (2·n_yᵢ)`). Optional train-only undersampling is also available. Class counts before and after this step are recorded in every result.

---

## 9. Evaluation metrics

Reported for every held-out run:
- Accuracy, Precision, Recall, F1 (positive class = Attack)
- Attack-specific Precision / Recall / F1
- False Positive Rate (FPR), False Negative Rate (FNR)
- Confusion matrix (TP / FP / TN / FN)
- Training time, inference time
- Support-vector count, sparsity percentage, primal objective value, solver status

High accuracy with weak attack recall is explicitly **not** treated as a successful detector. Decision scores are `wᵀx + b` — raw margin values, not probabilities.

---

## 10. Baseline comparison

Running the experiment mode retrains on the **identical data split** three ways:
- SVM + L1 with the configured λ
- The same QP with λ = 0 (plain L2 soft-margin SVM, no sparsity)
- A small `(C, λ)` hyperparameter grid (capped at 9 combinations)

The code does not assume L1 is automatically better — compare attack recall against sparsity yourself from the output.

---

## 11. Hyperparameter and threshold selection

- Hyperparameters (`C`, `λ`) are selected using **training/validation data only**
- The decision threshold is calibrated on the **validation split only**
- The final held-out test set is used **only** for the final reported evaluation — never for tuning

---

## 12. Actual results (real CIC-IDS2017 run)

| Metric | Value |
|---|---|
| Accuracy | 97.06% |
| Attack Recall | 97.78% |
| Attack F1 | 93.74% |
| Sparsity | 2.56% (76 of 78 features retained) |
| Selected hyperparameters | C = 1.0, λ(L1) = 0.2 |
| Solver status | solved |
| Objective value | 646.42 |

(Compared against an earlier SVM + NSL-KDD baseline attempt, which achieved 65% attack recall — this run shows a substantial improvement, on a different and more modern dataset.)

---

## 13. Project structure

```
module1/
├── data/            # dataset loading (CIC-IDS2017 + synthetic)
├── preprocessing/   # cleaning, splitting, imbalance handling
├── optimization/    # QP formulation + OSQP solver
├── analysis/        # KKT diagnostics, sparsity, support vectors
├── prediction/       # inference
├── evaluation/      # metrics + threshold tuning
├── experiments/     # SVM+L1 vs plain SVM baseline
├── models/          # API data schemas
├── services/        # orchestration layer (nid_service.py)
├── api/             # FastAPI app and routes
├── config/          # YAML configuration + settings
├── tests/           # 17 unit/integration tests
└── cli.py           # command-line entry point
```

---

## 14. Setup

```bash
pip install -r requirements.txt
```

## 15. Running

**Train on real CIC-IDS2017 data:**
```bash
python -m module1.cli train --source cicids2017 --out artifacts/cicids2017_train.json
```

**Train on synthetic data (quick smoke test, no dataset required):**
```bash
python -m module1.cli train --source synthetic --out artifacts/last_train.json
```

**Run the baseline comparison experiment:**
```bash
python -m module1.cli experiment --source cicids2017 --out artifacts/cicids2017_experiment.json
```

**Start the API server:**
```bash
python -m module1.cli serve
```
Interactive API docs: `http://localhost:8000/docs`
Health check: `http://localhost:8000/health`

## 16. API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Process liveness check |
| GET | `/api/module1/status` | Current state: idle / training / ready / error |
| POST | `/api/module1/train` | Train the model (`dataset_source`: `"cicids2017"` or `"synthetic"`) |
| POST | `/api/module1/predict` | Classify new traffic samples |
| POST | `/api/module1/evaluate` | Retrieve last held-out evaluation metrics |
| POST | `/api/module1/experiment` | Run baseline + hyperparameter grid |
| GET | `/api/module1/model-info` | Model, dataset, and optimisation summary |
| GET | `/api/module1/features` | Ranked feature weights |
| GET | `/api/module1/kkt` | Full KKT diagnostics report |
| GET | `/api/module1/metrics` | Evaluation metrics + sparsity |

Machine-readable schema: `GET /openapi.json` (no HTML UI).

## 17. Testing

```bash
python -m pytest
```

17 tests covering dataset loading and labelling, preprocessing, train/test splitting, feature scaling, the L1 QP formulation, solver output, prediction, evaluation metrics, sparsity analysis, KKT diagnostics, API endpoints, and error handling — all using small synthetic data tables so tests run instantly without the real dataset.

## 18. Reproducibility

Every result records: random seed, `C`, `λ`, data split, evaluation protocol, dataset source, and software versions (`numpy`, `pandas`, `scikit-learn`, `scipy`, `osqp`). OSQP is deterministic for a fixed problem, though residuals can vary slightly with solver tolerances.

## 19. Limitations

- Not production-ready; does not detect all attack types
- CIC-IDS2017 is a 2017 benchmark — real-world and current traffic patterns differ (encryption, topology, services)
- Concept drift: attacker and benign behaviour evolve after the dataset's capture period
- Only flow-level statistics are used; encrypted payload contents are not inspected
- Previously unseen attack types may be misclassified as benign or as a different known attack
- Dataset artefacts and class imbalance can inflate apparent accuracy
- The exact QP includes one slack variable per training row, so training on the full multi-million-row dataset may require sampling — any sampling used must be explicitly reported
- A linear decision boundary cannot capture all non-linear relationships in traffic data
- Feature weights indicate statistical association, not causation

Production deployment would require organisation-specific validation, ongoing monitoring, and an agreed operational false-positive budget — all outside the scope of this module.

## 20. Confirmation of scope

This module contains Python backend code, YAML configuration, automated tests, and documentation only. It does **not** contain a dashboard, React/HTML/CSS frontend, or any chart-rendering UI. A future OPTIVION frontend is expected to consume this module's JSON API directly rather than reimplementing any of the optimisation logic.