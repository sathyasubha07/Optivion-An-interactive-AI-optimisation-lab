"""Module registry. Adding a new module = one adapter + one entry in MODULES.

Everything the frontend shows about a module (cards, stepper text, maths, upload rules,
samples) comes from its entry here. Keys starting with "_" are server-only.
"""

from __future__ import annotations

from typing import Any

from backend.adapters import intrusion, satellite
from backend.adapters.base import Adapter

TEAM = {
    "project": "OptiVision — An Interactive AI Optimisation Lab",
    "course": "MFC course project — Team 08",
    "github": "https://github.com/sathyasubha07/Optivion-An-interactive-AI-optimisation-lab",
}

SATELLITE = {
    "id": "satellite-sr",
    "number": 4,
    "name": "Satellite Super-Resolution",
    "category": "Imaging",
    "tagline": "Sharpen a low-resolution satellite view by solving a sparse inverse problem.",
    "short_description": "Recover a sharper satellite image by solving WLS + L1 with ISTA.",
    "status": "ready",
    "method": "Weighted least squares + L1 sparsity, solved with ISTA",
    "method_chips": ["ISTA", "WLS", "L1", "KKT"],
    "input_type": "image",
    "accepted_file_types": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
    "max_upload_mb": 12,
    "input_help": "Any satellite or aerial image. It is converted to grayscale and resized to 128×128 as ground truth.",
    "problem": (
        "A low-resolution satellite observation is a blurred, downsampled version of an "
        "unknown high-resolution scene. Recovering that scene is ill-posed: many sharp "
        "images can explain the same pixels. The module treats the upload as ground truth, "
        "synthesises the corresponding low-resolution observation, then reconstructs it."
    ),
    "why_this_method": (
        "Super-resolution is ill-posed. The L1 term favours sparse edge structure and "
        "suppresses noise. ISTA handles the non-smooth L1 term with soft-thresholding and "
        "the x ≥ 0 constraint with projection. The method is matrix-free, so it never builds "
        "the 4096×16384 operator W and still scales. KKT residuals give a mathematical "
        "optimality check, and every ISTA iteration is recorded and inspectable."
    ),
    "alternatives_compared": [
        {"name": "Bicubic interpolation", "what": "A fixed upsampling kernel with no data fitting.",
         "why_not": "Cannot recover detail lost to blur and downsampling. It is the baseline, not the solver."},
        {"name": "Plain gradient descent", "what": "Smooth first-order descent on a differentiable loss.",
         "why_not": "Cannot handle the non-differentiable L1 term."},
        {"name": "CVXOPT / generic QP solvers", "what": "Interior-point solvers on an explicit quadratic program.",
         "why_not": "Impractical here: 16,384 variables and W would have to be materialised."},
        {"name": "Deep SR (e.g. SRCNN)", "what": "A trained convolutional mapping from LR to HR.",
         "why_not": "Needs large labelled training sets and gives no optimality certificate."},
    ],
    "math": {
        "objective": r"\min_{x \ge 0}\ \|Wx - y\|_2^2 + \lambda \|x\|_1",
        "constraints": [r"x \ge 0"],
        "update": {
            "label": "ISTA update",
            "latex": r"x^{k+1} = \max\!\big(0,\ S_{\lambda\alpha}\big(x^k - \alpha\, 2W^\top(Wx^k - y)\big)\big),\quad \alpha = 1/L",
        },
        "symbols": [
            {"symbol": r"y", "meaning": "Vectorised low-resolution observation (n = 64×64 = 4096)."},
            {"symbol": r"x", "meaning": "Unknown high-resolution image (N = 128×128 = 16,384)."},
            {"symbol": r"W", "meaning": "Degradation operator: Gaussian blur then downsample. Applied matrix-free as Wx and Wᵀr."},
            {"symbol": r"\lambda", "meaning": "L1 weight. Larger λ favours sparser reconstructions."},
            {"symbol": r"S_\tau(z)", "meaning": "Soft-threshold sign(z)·max(|z| − τ, 0), the proximal operator of the L1 norm."},
            {"symbol": r"L", "meaning": "Lipschitz constant of the data-term gradient, estimated by power iteration."},
        ],
        "dimensions": "W is n×N with n = 4096 and N = 16,384 at the default 2× scale.",
    },
    "algorithm_steps": [
        "Load and normalise the upload; resize it to a 128×128 high-resolution ground truth.",
        "Synthesise y = downsample(blur(x_gt)).",
        "Initialise x with a bicubic upsample of y.",
        "Estimate the Lipschitz constant of ∇‖Wx − y‖² by power iteration (matrix-free).",
        "ISTA: gradient step on the data term, soft-threshold for L1, project onto x ≥ 0.",
        "Stop when the relative objective change falls below tolerance, or at the iteration cap.",
        "Evaluate PSNR/SSIM against ground truth and the bicubic baseline; check KKT residuals.",
    ],
    "_adapter": satellite.ADAPTER,
}

INTRUSION = {
    "id": "network-intrusion",
    "number": 1,
    "name": "Network Intrusion Detection",
    "category": "Cybersecurity",
    "tagline": "Separate attack traffic from benign flows with a sparse, certifiable SVM.",
    "short_description": "Detect attack vs benign flows with a soft-margin SVM + L1, solved as a QP.",
    "status": "ready",
    "method": "Soft-margin linear SVM + L1 sparsity, convex QP solved with OSQP",
    "method_chips": ["QP", "SVM", "L1", "KKT", "OSQP"],
    "input_type": "csv",
    "accepted_file_types": [".csv"],
    "max_upload_mb": 50,
    "input_help": (
        "A flow-statistics CSV in CIC-IDS2017 layout: numeric feature columns plus a 'Label' column "
        "(BENIGN or an attack name). Large files are capped at 20,000 rows and a 3,000-row stratified sample."
    ),
    "problem": (
        "Given flow-level network statistics, classify each connection as Benign or Attack. "
        "The module trains a linear SVM with an L1 penalty as a convex quadratic program and "
        "reports which features it relies on, which training flows are support vectors, and "
        "whether the solution satisfies the KKT optimality conditions. Its intended dataset is "
        "CIC-IDS2017; the sample uses the module's synthetic flows because no real dataset is bundled."
    ),
    "why_this_method": (
        "A linear SVM is a convex QP with a margin and dual multipliers that identify support "
        "vectors, so KKT residuals are meaningful rather than decorative. L1 drives many feature "
        "weights to zero, showing which flow statistics the detector actually uses. OSQP returns "
        "primal/dual residuals and iteration counts instead of hiding inside sklearn.svm.SVC. "
        "The module does not claim to beat every modern IDS; it claims an inspectable optimum."
    ),
    "alternatives_compared": [
        {"name": "Logistic regression", "what": "Smooth convex classification via Newton or gradient methods.",
         "why_not": "No geometric margin or support-vector set to inspect."},
        {"name": "SGD-trained SVM (Pegasos)", "what": "Stochastic subgradient SVM, fast on huge data.",
         "why_not": "Approximate: no single certified QP solution and no clean dual multipliers."},
        {"name": "Random forest", "what": "Non-linear tree ensemble, often strong on tabular IDS data.",
         "why_not": "No convex optimality certificate, and importances are not sparse weights."},
        {"name": "Deep neural nets", "what": "High-capacity black-box classifiers.",
         "why_not": "Need large data, hide the optimisation path, and give no KKT certificate."},
    ],
    "math": {
        "objective": r"\min_{w,b,\xi}\ \tfrac12\|w\|_2^2 + C\sum_i \xi_i + \lambda \|w\|_1",
        "constraints": [r"y_i\,(w^\top x_i + b) \ge 1 - \xi_i", r"\xi_i \ge 0"],
        "update": {
            "label": "Split-variable QP (what OSQP actually solves)",
            "latex": r"w = w^+ - w^-,\quad w^\pm \ge 0,\quad \|w\|_1 = \mathbf{1}^\top(w^+ + w^-)",
        },
        "symbols": [
            {"symbol": r"x_i", "meaning": "Feature vector of flow i after train-only scaling."},
            {"symbol": r"y_i \in \{-1,+1\}", "meaning": "Label: Attack = +1, Benign = −1."},
            {"symbol": r"w, b", "meaning": "Linear weights and bias. Decision score wᵀx + b is a margin, not a probability."},
            {"symbol": r"\xi_i", "meaning": "Slack: allows misclassified or in-margin points."},
            {"symbol": r"C", "meaning": "Soft-margin trade-off (class-weighted per sample when balancing is on)."},
            {"symbol": r"\lambda", "meaning": "L1 sparsity weight."},
        ],
        "dimensions": "Variables: 2d for w± plus one slack per training row plus the bias.",
    },
    "algorithm_steps": [
        "Load the CSV and map BENIGN → 0, any other label → 1.",
        "Clean, drop identifiers and duplicates; stratified train / validation / test split.",
        "Impute and scale using the training split only; class-weighted C_i for imbalance.",
        "Select C, λ and the decision threshold on the validation split only.",
        "Build the split-variable soft-margin + L1 QP and solve it with OSQP.",
        "Read KKT residuals, weight sparsity and support vectors from the QP solution.",
        "Evaluate on the held-out test split and explain each prediction by wⱼxⱼ contributions.",
    ],
    "_adapter": intrusion.ADAPTER,
}


MODULES: list[dict[str, Any]] = [INTRUSION, SATELLITE]


def all_modules() -> list[dict[str, Any]]:
    return MODULES


def get_module(module_id: str) -> dict[str, Any] | None:
    return next((m for m in MODULES if m["id"] == module_id), None)


def get_adapter(entry: dict[str, Any]) -> Adapter | None:
    return entry.get("_adapter")


def public_module(entry: dict[str, Any]) -> dict[str, Any]:
    data = {k: v for k, v in entry.items() if not k.startswith("_")}
    adapter = get_adapter(entry)
    data["samples"] = [s.as_dict() for s in adapter.list_samples()] if adapter else []
    return data
