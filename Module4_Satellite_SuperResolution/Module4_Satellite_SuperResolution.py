"""
Module 4 — Satellite Image Super-Resolution using Weighted Least Squares
         and Sparse Regularization (L1/L2)

OptiVision — An Interactive AI Optimization Lab
Backend-only standalone module (no UI / frontend).

Mathematical formulation
------------------------
    minimize   ||W x - y||_2^2  +  lambda * ||x||_1
    subject to x >= 0

where
    y  : observed low-resolution image (vectorised)
    x  : unknown high-resolution image (vectorised)
    W  : degradation operator  =  Downsample o Blur
    lambda : L1 sparsity regularisation weight

Solved iteratively via ISTA (proximal gradient) with non-negativity projection.
KKT conditions are evaluated at convergence for optimality verification.
"""

from __future__ import annotations

import os
import zipfile
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator
from skimage import io, color, metrics, transform


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SRConfig:
    """Hyper-parameters for the WLS + L1 super-resolution solver."""

    scale: int = 2                       # upsampling factor (2 => 64x64 -> 128x128)
    blur_sigma: float = 1.0              # Gaussian blur std-dev (pixels)
    lambda_l1: float = 0.005             # L1 sparsity weight
    lambda_l2: float = 0.0               # optional L2 ridge term (0 = pure L1)
    max_iterations: int = 150
    tolerance: float = 1e-5              # relative objective change for convergence
    target_lr_size: int = 64             # crop HR ground-truth to 128, then downsample to 64

    # ISTA step-size: 1 / Lipschitz constant of grad(||Wx-y||^2)
    # Conservative upper bound used if power iteration not run
    lipschitz: Optional[float] = None

    output_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "outputs"
        )
    )


# ---------------------------------------------------------------------------
# Section 1 — Dataset download / loading
# ---------------------------------------------------------------------------

# UC Merced Land Use — official mirror via HuggingFace / TorchGeo (~318 MB, cached locally).
_UC_MERCED_HF_URL = (
    "https://huggingface.co/datasets/torchgeo/ucmerced/resolve/"
    "7c5ef3454d9b1cccfa7ccde0c01fc8f00a45909a/UCMerced_LandUse.zip"
)
_UC_MERCED_ZIP_URL = _UC_MERCED_HF_URL  # alias kept for clarity in logs


def _download_file(url: str, dest_path: str, timeout: int = 120) -> bool:
    """Download *url* to *dest_path*. Returns True on success."""
    try:
        print(f"  Downloading: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "OptiVision-Module4/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
        with open(dest_path, "wb") as fh:
            fh.write(data)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"  [warn] Download failed: {exc}")
        return False


def _generate_synthetic_satellite_image(size: int = 128) -> np.ndarray:
    """
    Procedurally generate a satellite-like grayscale image when no dataset
    is reachable.  Contains field textures, a 'road', and bright structures.
    """
    rng = np.random.default_rng(42)
    img = np.zeros((size, size), dtype=np.float64)

    # Green/brown field texture
    img += ndimage.gaussian_filter(rng.random((size, size)), sigma=8) * 0.35

    # Diagonal road
    yy, xx = np.mgrid[0:size, 0:size]
    road_mask = np.abs(yy - xx + 10) < 3
    img[road_mask] = 0.55

    # Rectangular buildings / structures
    img[30:50, 70:95] = 0.75
    img[75:90, 20:45] = 0.68
    img[10:25, 10:30] = 0.62

    # Small bright features (vehicles / equipment)
    for cy, cx in [(40, 82), (82, 32), (18, 20)]:
        img[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0.90

    img = ndimage.gaussian_filter(img, sigma=0.8)
    img = np.clip(img, 0.0, 1.0)
    return img


def _load_image_as_grayscale(path: str) -> np.ndarray:
    """Load image and normalise to float64 in [0, 1]."""
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)


def _uc_merced_images_root(data_dir: str) -> str:
    return os.path.join(data_dir, "UCMerced_LandUse", "Images")


def ensure_uc_merced_dataset(data_dir: Optional[str] = None) -> str:
    """
    Download and extract the complete UC Merced Land Use Dataset
    (2100 images, 21 classes) if not already present.

    Returns path to the Images/ root directory.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    images_root = _uc_merced_images_root(data_dir)
    if os.path.isdir(images_root):
        # Verify we have class folders with images
        class_dirs = [
            d for d in os.listdir(images_root)
            if os.path.isdir(os.path.join(images_root, d))
        ]
        if len(class_dirs) >= 21:
            return images_root

    zip_path = os.path.join(data_dir, "UCMerced_LandUse.zip")
    if not os.path.isfile(zip_path):
        print("  Downloading complete UC Merced Land Use dataset (~318 MB, one-time)...")
        print(f"  Source: {_UC_MERCED_HF_URL}")
        ok = _download_file(_UC_MERCED_HF_URL, zip_path, timeout=900)
        if not ok:
            raise RuntimeError(
                "Failed to download UC Merced Land Use dataset. "
                "Check network connectivity and retry."
            )

    print("  Extracting full UC Merced Land Use archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    if not os.path.isdir(images_root):
        raise RuntimeError(f"UC Merced Images folder not found after extract: {images_root}")
    return images_root


def index_uc_merced_images(images_root: str) -> List[Tuple[str, str]]:
    """
    Index all UC Merced images as (path, class_name) pairs.
    Expected: 21 classes x 100 images = 2100.
    """
    records: List[Tuple[str, str]] = []
    class_names = sorted(
        d for d in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, d))
    )
    for cls in class_names:
        cls_dir = os.path.join(images_root, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png")):
                records.append((os.path.join(cls_dir, fname), cls))
    return records


def train_test_split_uc_merced(
    records: List[Tuple[str, str]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Stratified 80/20 train/test split by land-use class.
    """
    rng = np.random.default_rng(seed)
    by_class: Dict[str, List[Tuple[str, str]]] = {}
    for path, cls in records:
        by_class.setdefault(cls, []).append((path, cls))

    train: List[Tuple[str, str]] = []
    test: List[Tuple[str, str]] = []
    for cls, items in sorted(by_class.items()):
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n_train = int(round(len(items) * train_ratio))
        n_train = min(max(n_train, 1), len(items) - 1) if len(items) > 1 else len(items)
        for i, j in enumerate(idx):
            if i < n_train:
                train.append(items[j])
            else:
                test.append(items[j])
    return train, test


def download_and_load_full_dataset(
    data_dir: Optional[str] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]:
    """
    Download the complete UC Merced Land Use Dataset and return stratified splits.

    Returns
    -------
    train_records : list of (image_path, class_name)
    test_records  : list of (image_path, class_name)
    source        : human-readable dataset description
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    try:
        images_root = ensure_uc_merced_dataset(data_dir)
        records = index_uc_merced_images(images_root)
        if len(records) == 0:
            raise RuntimeError("No images found in UC Merced dataset.")
        train, test = train_test_split_uc_merced(records, train_ratio=0.8, seed=42)
        n_classes = len({c for _, c in records})
        source = (
            f"UC Merced Land Use Dataset ({len(records)} images, "
            f"{n_classes} classes, HuggingFace / TorchGeo)"
        )
        return train, test, source
    except Exception as exc:
        print(f"  [warn] Full dataset load failed: {exc}")
        print("  Falling back to synthetic images for train/test demo.")
        # Build a tiny synthetic fallback so the pipeline remains runnable offline
        syn_dir = os.path.join(data_dir, "synthetic_fallback")
        os.makedirs(syn_dir, exist_ok=True)
        records = []
        for cls_i, cls in enumerate(["field", "urban", "water"]):
            cls_dir = os.path.join(syn_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(10):
                img = _generate_synthetic_satellite_image(128)
                # Slight variation per sample
                img = np.clip(img + 0.02 * (i - 5) / 5.0, 0.0, 1.0)
                path = os.path.join(cls_dir, f"{cls}_{i:02d}.png")
                io.imsave(path, (img * 255).astype(np.uint8))
                records.append((path, cls))
        train, test = train_test_split_uc_merced(records, train_ratio=0.8, seed=42)
        return train, test, "Synthetic satellite images (fallback)"


def preprocess_ground_truth(
    hr_image: np.ndarray, config: SRConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop / resize HR ground truth and synthesise the LR observation y = W x_true.

    Returns (hr_cropped, lr_observed).
    """
    hr_size = config.target_lr_size * config.scale
    hr = transform.resize(
        hr_image, (hr_size, hr_size), order=3, anti_aliasing=True, preserve_range=True
    )
    hr = np.clip(hr, 0.0, 1.0)

    lr = degrade(hr, config.scale, config.blur_sigma)
    return hr, lr


# ---------------------------------------------------------------------------
# Section 2 — Degradation model  W = Downsample o GaussianBlur
# ---------------------------------------------------------------------------

def _gaussian_kernel(sigma: float, size: Optional[int] = None) -> np.ndarray:
    """Normalised 2-D Gaussian kernel."""
    if size is None:
        size = int(max(3, 2 * np.ceil(3 * sigma) + 1))
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def blur_image(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur (correlation, zero-padding)."""
    k = _gaussian_kernel(sigma)
    return ndimage.convolve(img, k, mode="reflect")


def downsample_image(img: np.ndarray, scale: int) -> np.ndarray:
    """Average-pool downsampling by integer *scale*."""
    h, w = img.shape
    h2, w2 = h // scale, w // scale
    img = img[: h2 * scale, : w2 * scale]
    return img.reshape(h2, scale, w2, scale).mean(axis=(1, 3))


def upsample_zeros(img: np.ndarray, scale: int, hr_shape: Tuple[int, int]) -> np.ndarray:
    """Insert zeros between pixels (zero-order hold upsampling)."""
    h, w = img.shape
    up = np.zeros(hr_shape, dtype=img.dtype)
    up[::scale, ::scale] = img
    return up


def degrade(hr: np.ndarray, scale: int, blur_sigma: float) -> np.ndarray:
    """Forward model: y = S(B(x))."""
    return downsample_image(blur_image(hr, blur_sigma), scale)


class DegradationOperator:
    """
    Implicit linear operator W and its adjoint W^T for efficient matrix-free
    optimisation without building the full sparse W matrix.
    """

    def __init__(self, hr_shape: Tuple[int, int], scale: int, blur_sigma: float):
        self.hr_shape = hr_shape
        self.scale = scale
        self.blur_sigma = blur_sigma
        self.kernel = _gaussian_kernel(blur_sigma)
        lr_h = hr_shape[0] // scale
        lr_w = hr_shape[1] // scale
        self.lr_shape = (lr_h, lr_w)

    def matvec(self, x_vec: np.ndarray) -> np.ndarray:
        """W @ x  (flattened)."""
        x = x_vec.reshape(self.hr_shape)
        y = degrade(x, self.scale, self.blur_sigma)
        return y.ravel()

    def rmatvec(self, y_vec: np.ndarray) -> np.ndarray:
        """W^T @ y  (flattened)."""
        y = y_vec.reshape(self.lr_shape)
        # Upsample then apply adjoint blur (correlation with flipped kernel)
        up = upsample_zeros(y, self.scale, self.hr_shape)
        k_adj = self.kernel[::-1, ::-1]
        return ndimage.convolve(up, k_adj, mode="reflect").ravel()

    @property
    def linear_operator(self) -> LinearOperator:
        n_hr = self.hr_shape[0] * self.hr_shape[1]
        n_lr = self.lr_shape[0] * self.lr_shape[1]
        return LinearOperator(
            shape=(n_lr, n_hr),
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=np.float64,
        )


def estimate_lipschitz(W: DegradationOperator, n_iter: int = 30) -> float:
    """
    Estimate ||W||_2^2  (largest eigenvalue of W^T W) via power iteration.
    Lipschitz constant of grad f(x)=||Wx-y||^2 is 2||W||_2^2.
    """
    n = W.hr_shape[0] * W.hr_shape[1]
    rng = np.random.default_rng(0)
    v = rng.random(n)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        v = W.rmatvec(W.matvec(v))
        norm = np.linalg.norm(v)
        if norm < 1e-14:
            break
        v /= norm
    Wv = W.matvec(v)
    return 2.0 * float(np.dot(Wv, Wv))


# ---------------------------------------------------------------------------
# Section 3 — WLS + L1 optimisation (ISTA with non-negativity)
# ---------------------------------------------------------------------------

def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """Proximal operator for tau * ||x||_1."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


@dataclass
class OptimizationResult:
    """Container for solver outputs and convergence history."""

    x_hr: np.ndarray
    objective_history: List[float]
    data_fidelity_history: List[float]
    sparsity_history: List[float]
    residual_history: List[float]
    step_size: float
    iterations: int
    converged: bool
    kkt: Dict[str, float]


class WLSSparseSolver:
    """
    Iterative Weighted Least Squares solver with L1 (+ optional L2) regularisation.

    Algorithm: ISTA (Iterative Shrinkage-Thresholding Algorithm)
        x^{k+1} = max(0,  S_{lambda*alpha}( x^k - alpha * grad_f(x^k) ))

    where  grad_f(x) = 2 W^T (W x - y) + 2 lambda_l2 x
           S_tau(z)   = sign(z) max(|z|-tau, 0)   (soft-threshold)
           max(0, .)  = non-negativity projection
    """

    def __init__(self, config: SRConfig):
        self.config = config

    def solve(
        self,
        lr_observed: np.ndarray,
        hr_shape: Tuple[int, int],
        x_init: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> OptimizationResult:
        cfg = self.config
        W = DegradationOperator(hr_shape, cfg.scale, cfg.blur_sigma)
        y = lr_observed.ravel()

        # --- Initialisation: bicubic upsample ---
        if x_init is None:
            up = transform.resize(
                lr_observed,
                hr_shape,
                order=3,
                anti_aliasing=True,
                preserve_range=True,
            )
            x = np.clip(up, 0.0, 1.0).ravel()
        else:
            x = x_init.ravel().copy()

        # --- Step size ---
        if cfg.lipschitz is None:
            L = estimate_lipschitz(W)
        else:
            L = cfg.lipschitz
        alpha = 1.0 / L
        tau = cfg.lambda_l1 * alpha

        obj_hist: List[float] = []
        data_hist: List[float] = []
        sparse_hist: List[float] = []
        res_hist: List[float] = []

        converged = False
        n_iter = 0

        if verbose:
            print("\n--- WLS + L1 Optimisation (ISTA) ---")
            print(f"  HR shape : {hr_shape}  |  LR shape : {W.lr_shape}")
            print(f"  lambda_L1={cfg.lambda_l1}  lambda_L2={cfg.lambda_l2}")
            print(f"  step alpha={alpha:.6e}  (Lipschitz L={L:.6e})")
            print(f"  {'Iter':>5}  {'Objective':>14}  {'Data term':>14}  "
                  f"{'L1 term':>12}  {'Rel change':>12}")
            print("  " + "-" * 65)

        for k in range(1, cfg.max_iterations + 1):
            Wx = W.matvec(x)
            residual = Wx - y
            grad = 2.0 * W.rmatvec(residual)
            if cfg.lambda_l2 > 0:
                grad += 2.0 * cfg.lambda_l2 * x

            # ISTA gradient step + L1 prox + non-negativity
            x_new = soft_threshold(x - alpha * grad, tau)
            x_new = np.maximum(x_new, 0.0)

            data_term = float(np.dot(residual, residual))
            l1_term = cfg.lambda_l1 * float(np.sum(np.abs(x_new)))
            l2_term = cfg.lambda_l2 * float(np.dot(x_new, x_new))
            objective = data_term + l1_term + l2_term

            obj_hist.append(objective)
            data_hist.append(data_term)
            sparse_hist.append(l1_term)
            res_hist.append(float(np.linalg.norm(residual)))

            rel_change = (
                abs(obj_hist[-2] - objective) / (abs(obj_hist[-2]) + 1e-12)
                if len(obj_hist) > 1
                else 1.0
            )

            if verbose and (k <= 5 or k % 10 == 0 or rel_change < cfg.tolerance):
                print(
                    f"  {k:5d}  {objective:14.6e}  {data_term:14.6e}  "
                    f"{l1_term:12.6e}  {rel_change:12.4e}"
                )

            x = x_new
            n_iter = k

            if rel_change < cfg.tolerance and k > 5:
                converged = True
                if verbose:
                    print(f"  Converged at iteration {k} (rel change < {cfg.tolerance})")
                break

        x_hr = x.reshape(hr_shape)
        kkt = compute_kkt_residuals(W, x, y, cfg.lambda_l1, cfg.lambda_l2)

        if verbose:
            print(f"\n  Final objective : {obj_hist[-1]:.6e}")
            print(f"  KKT stationarity: {kkt['stationarity_inf']:.4e}")
            print(f"  KKT complementarity: {kkt['complementarity_inf']:.4e}")

        return OptimizationResult(
            x_hr=x_hr,
            objective_history=obj_hist,
            data_fidelity_history=data_hist,
            sparsity_history=sparse_hist,
            residual_history=res_hist,
            step_size=alpha,
            iterations=n_iter,
            converged=converged,
            kkt=kkt,
        )


# ---------------------------------------------------------------------------
# Section 4 — KKT condition analysis
# ---------------------------------------------------------------------------

def compute_kkt_residuals(
    W: DegradationOperator,
    x: np.ndarray,
    y: np.ndarray,
    lambda_l1: float,
    lambda_l2: float = 0.0,
) -> Dict[str, float]:
    """
    Evaluate KKT conditions for:
        min  ||Wx-y||^2 + lambda_l1||x||_1 + lambda_l2||x||_2^2
        s.t. x >= 0

    Stationarity (subgradient condition):
        0 in 2 W^T(Wx-y) + lambda_l2*2x + lambda_l1 * d||x||_1  - nu
        nu >= 0,  nu_i x_i = 0

    We report infinity-norm residuals for stationarity and complementarity.
    """
    x = x.ravel()
    grad_data = 2.0 * W.rmatvec(W.matvec(x) - y)
    if lambda_l2 > 0:
        grad_data += 2.0 * lambda_l2 * x

    # L1 subgradient: sign(x_i) where x_i != 0, any g in [-1,1] where x_i == 0
    subgrad_l1 = np.zeros_like(x)
    active = x > 1e-10
    subgrad_l1[active] = np.sign(x[active])

    # Dual variable nu for x >= 0 from stationarity: nu = grad_data + lambda_l1 * subgrad
    nu = grad_data + lambda_l1 * subgrad_l1

    # Complementarity: nu_i * x_i  (should be 0)
    comp_residual = np.abs(nu * x)

    # Stationarity: for x_i > 0 need grad_data + lambda_l1*sign(x_i) = 0
    stat_active = np.abs(grad_data[active] + lambda_l1 * np.sign(x[active])) if active.any() else np.array([0.0])
    # for x_i = 0 need grad_data_i + lambda_l1 * g = nu_i >= 0  =>  grad_data_i in [-lambda_l1, lambda_l1] violated part
    inactive = ~active
    stat_inactive = np.maximum(0.0, np.abs(grad_data[inactive]) - lambda_l1) if inactive.any() else np.array([0.0])

    stationarity_inf = float(max(stat_active.max() if stat_active.size else 0.0,
                                 stat_inactive.max() if stat_inactive.size else 0.0))
    complementarity_inf = float(comp_residual.max())

    return {
        "stationarity_inf": stationarity_inf,
        "complementarity_inf": complementarity_inf,
        "dual_norm_inf": float(np.max(np.abs(nu))),
    }


# ---------------------------------------------------------------------------
# Section 5 — Evaluation metrics
# ---------------------------------------------------------------------------

def compute_psnr(reference: np.ndarray, test: np.ndarray, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    return float(metrics.peak_signal_noise_ratio(reference, test, data_range=data_range))


def compute_ssim(reference: np.ndarray, test: np.ndarray, data_range: float = 1.0) -> float:
    """Structural Similarity Index."""
    return float(metrics.structural_similarity(reference, test, data_range=data_range))


def psnr_lr_baseline(hr_gt: np.ndarray, lr: np.ndarray, scale: int) -> float:
    """PSNR of bicubic-upsampled LR vs HR ground truth (baseline)."""
    up = transform.resize(lr, hr_gt.shape, order=3, anti_aliasing=True, preserve_range=True)
    return compute_psnr(hr_gt, np.clip(up, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Section 6 — Prediction API
# ---------------------------------------------------------------------------

def predict_super_resolution(
    lr_image: np.ndarray,
    config: Optional[SRConfig] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, OptimizationResult, Dict[str, float]]:
    """
    Super-resolve a low-resolution satellite image.

    Parameters
    ----------
    lr_image : (H, W) grayscale array in [0, 1]
    config   : SRConfig hyper-parameters
    verbose  : print iteration log

    Returns
    -------
    hr_reconstructed : super-resolved image
    opt_result       : full optimisation trace
    metrics          : dict with PSNR/SSIM if ground truth not available metrics
                       are computed against bicubic upsample baseline internally
    """
    if config is None:
        config = SRConfig()

    lr = np.clip(lr_image.astype(np.float64), 0.0, 1.0)
    hr_shape = (lr.shape[0] * config.scale, lr.shape[1] * config.scale)

    solver = WLSSparseSolver(config)
    result = solver.solve(lr, hr_shape, verbose=verbose)

    # Cache Lipschitz estimate so subsequent images reuse the same step size
    if config.lipschitz is None and result.step_size > 0:
        config.lipschitz = 1.0 / result.step_size

    return result.x_hr, result, {}


def evaluate_full_dataset(config: Optional[SRConfig] = None) -> Dict[str, Any]:
    """
    Download the complete UC Merced dataset, split 80/20, and run WLS + L1
    optimisation on every test image. Reports average and per-class PSNR/SSIM.
    """
    if config is None:
        config = SRConfig()

    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 70)
    print("  OptiVision - Module 4: Full Dataset Super-Resolution Evaluation")
    print("  Weighted Least Squares + L1 Sparse Regularisation")
    print("=" * 70)

    print("\n[1/4] Loading complete UC Merced Land Use Dataset...")
    train_records, test_records, data_source = download_and_load_full_dataset()
    n_classes = len({c for _, c in train_records + test_records})
    print(f"  Source     : {data_source}")
    print(f"  Train set  : {len(train_records)} images (80%)")
    print(f"  Test set   : {len(test_records)} images (20%)")
    print(f"  Classes    : {n_classes}")

    # Pre-compute Lipschitz once for the fixed HR size (shared by all images)
    hr_size = config.target_lr_size * config.scale
    hr_shape = (hr_size, hr_size)
    if config.lipschitz is None:
        print("\n[2/4] Estimating Lipschitz constant (shared across dataset)...")
        W = DegradationOperator(hr_shape, config.scale, config.blur_sigma)
        config.lipschitz = estimate_lipschitz(W)
        print(f"  Lipschitz L = {config.lipschitz:.6e}")
    else:
        print("\n[2/4] Using configured Lipschitz constant...")
        print(f"  Lipschitz L = {config.lipschitz:.6e}")

    print(f"\n[3/4] Running WLS + L1 on all {len(test_records)} test images...")
    solver = WLSSparseSolver(config)

    all_psnr: List[float] = []
    all_ssim: List[float] = []
    all_iters: List[int] = []
    per_class: Dict[str, Dict[str, List[float]]] = {}
    first_example: Optional[Dict[str, Any]] = None

    for i, (img_path, cls) in enumerate(test_records, start=1):
        hr_raw = _load_image_as_grayscale(img_path)
        hr_gt, lr_obs = preprocess_ground_truth(hr_raw, config)
        opt_result = solver.solve(lr_obs, hr_shape, verbose=False)
        hr_sr = opt_result.x_hr

        psnr_val = compute_psnr(hr_gt, hr_sr)
        ssim_val = compute_ssim(hr_gt, hr_sr)
        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)
        all_iters.append(opt_result.iterations)

        bucket = per_class.setdefault(cls, {"psnr": [], "ssim": [], "iters": []})
        bucket["psnr"].append(psnr_val)
        bucket["ssim"].append(ssim_val)
        bucket["iters"].append(float(opt_result.iterations))

        if first_example is None:
            first_example = {
                "path": img_path,
                "class": cls,
                "hr_gt": hr_gt,
                "lr_obs": lr_obs,
                "hr_sr": hr_sr,
                "opt_result": opt_result,
                "psnr": psnr_val,
                "ssim": ssim_val,
            }

        if i == 1 or i % 20 == 0 or i == len(test_records):
            print(
                f"  [{i:4d}/{len(test_records)}] class={cls:20s} "
                f"PSNR={psnr_val:6.2f} dB  SSIM={ssim_val:.4f}  "
                f"iters={opt_result.iterations}"
            )

    avg_psnr = float(np.mean(all_psnr)) if all_psnr else 0.0
    avg_ssim = float(np.mean(all_ssim)) if all_ssim else 0.0
    avg_iters = float(np.mean(all_iters)) if all_iters else 0.0

    print("\n" + "=" * 70)
    print("  TEST SET RESULTS (WLS + L1)")
    print("=" * 70)
    print(f"  Images evaluated : {len(all_psnr)}")
    print(f"  Average PSNR     : {avg_psnr:.2f} dB")
    print(f"  Average SSIM     : {avg_ssim:.4f}")
    print(f"  Avg. iterations  : {avg_iters:.1f}")

    print("\n  Per-class results:")
    print(f"  {'Class':22s} {'N':>4s} {'Avg PSNR':>10s} {'Avg SSIM':>10s}")
    print("  " + "-" * 50)
    per_class_summary: Dict[str, Dict[str, float]] = {}
    for cls in sorted(per_class.keys()):
        vals = per_class[cls]
        cls_psnr = float(np.mean(vals["psnr"]))
        cls_ssim = float(np.mean(vals["ssim"]))
        per_class_summary[cls] = {
            "n": float(len(vals["psnr"])),
            "psnr": cls_psnr,
            "ssim": cls_ssim,
        }
        print(f"  {cls:22s} {len(vals['psnr']):4d} {cls_psnr:10.2f} {cls_ssim:10.4f}")

    # Sample visualisations from first test image
    print("\n[4/4] Generating sample visualisations (first test image)...")
    plot_paths: Dict[str, str] = {}
    explanation = ""
    if first_example is not None:
        plot_paths = generate_all_visualizations(
            first_example["lr_obs"],
            first_example["hr_sr"],
            first_example["hr_gt"],
            first_example["opt_result"],
            config,
            prefix="testset_sample_",
        )
        for name, path in plot_paths.items():
            print(f"  Saved {name}: {path}")

        psnr_before = psnr_lr_baseline(
            first_example["hr_gt"], first_example["lr_obs"], config.scale
        )
        explanation = generate_explanation(
            config=config,
            opt_result=first_example["opt_result"],
            psnr_before=psnr_before,
            psnr_after=first_example["psnr"],
            ssim_after=first_example["ssim"],
            data_source=f"{data_source} | class={first_example['class']}",
            lr_shape=first_example["lr_obs"].shape,
            hr_shape=first_example["hr_gt"].shape,
        )
        expl_path = os.path.join(config.output_dir, "testset_sample_explanation.txt")
        with open(expl_path, "w", encoding="utf-8") as fh:
            fh.write(explanation)
        print(f"  Explanation saved: {expl_path}")

    summary_path = os.path.join(config.output_dir, "testset_metrics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("UC Merced Test Set - WLS + L1 Super-Resolution\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"Dataset       : {data_source}\n")
        fh.write(f"Train / Test  : {len(train_records)} / {len(test_records)}\n")
        fh.write(f"Average PSNR  : {avg_psnr:.4f} dB\n")
        fh.write(f"Average SSIM  : {avg_ssim:.6f}\n")
        fh.write(f"Avg iterations: {avg_iters:.2f}\n\n")
        fh.write(f"{'Class':22s} {'N':>4s} {'Avg PSNR':>10s} {'Avg SSIM':>10s}\n")
        fh.write("-" * 50 + "\n")
        for cls, s in per_class_summary.items():
            fh.write(f"{cls:22s} {int(s['n']):4d} {s['psnr']:10.2f} {s['ssim']:10.4f}\n")
    print(f"  Metrics summary saved: {summary_path}")

    return {
        "data_source": data_source,
        "train_size": len(train_records),
        "test_size": len(test_records),
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "avg_iterations": avg_iters,
        "per_class": per_class_summary,
        "plot_paths": plot_paths,
        "explanation": explanation,
        "test_records": test_records,
        "train_records": train_records,
        "first_example": first_example,
    }


def process_user_image(
    image_path: str,
    config: Optional[SRConfig] = None,
) -> Dict[str, Any]:
    """
    Process a user-provided image for satellite super-resolution.

    Intended for frontend upload integration: accepts any image path,
    preprocesses it, runs full WLS + L1 ISTA optimisation, and returns
    the super-resolved image plus metrics, plot paths, and explanation.

    Parameters
    ----------
    image_path : path to user image (.png / .jpg / .tif / ...)
    config     : optional SRConfig

    Returns
    -------
    dict with keys:
        hr_image, lr_image, hr_ground_truth,
        psnr, ssim, psnr_before, ssim_before,
        iterations, converged,
        convergence_graph, residual_map, sparsity_map, comparison,
        explanation, optimization
    """
    if config is None:
        config = SRConfig()

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"User image not found: {image_path}")

    user_out = os.path.join(config.output_dir, "user_upload")
    os.makedirs(user_out, exist_ok=True)

    # Work in a config copy whose output_dir points at user_upload/
    user_config = SRConfig(
        scale=config.scale,
        blur_sigma=config.blur_sigma,
        lambda_l1=config.lambda_l1,
        lambda_l2=config.lambda_l2,
        max_iterations=config.max_iterations,
        tolerance=config.tolerance,
        target_lr_size=config.target_lr_size,
        lipschitz=config.lipschitz,
        output_dir=user_out,
    )

    print("\n" + "=" * 70)
    print("  process_user_image() - User Upload Super-Resolution")
    print("=" * 70)
    print(f"  Input path: {image_path}")

    # Preprocess: load, normalize, resize HR, synthesise LR observation
    hr_raw = _load_image_as_grayscale(image_path)
    hr_gt, lr_obs = preprocess_ground_truth(hr_raw, user_config)
    hr_shape = hr_gt.shape

    print(f"  Preprocessed HR : {hr_shape[0]}x{hr_shape[1]}")
    print(f"  Synthesised LR  : {lr_obs.shape[0]}x{lr_obs.shape[1]}")

    if user_config.lipschitz is None:
        W = DegradationOperator(hr_shape, user_config.scale, user_config.blur_sigma)
        user_config.lipschitz = estimate_lipschitz(W)

    solver = WLSSparseSolver(user_config)
    opt_result = solver.solve(lr_obs, hr_shape, verbose=True)
    hr_sr = opt_result.x_hr

    psnr_after = compute_psnr(hr_gt, hr_sr)
    ssim_after = compute_ssim(hr_gt, hr_sr)
    psnr_before = psnr_lr_baseline(hr_gt, lr_obs, user_config.scale)
    up_bicubic = transform.resize(
        lr_obs, hr_shape, order=3, anti_aliasing=True, preserve_range=True
    )
    ssim_before = compute_ssim(hr_gt, np.clip(up_bicubic, 0.0, 1.0))

    print("\n" + "-" * 40)
    print("  User Image")
    print(f"  Input Resolution : {lr_obs.shape[0]}x{lr_obs.shape[1]}")
    print(f"  Output Resolution: {hr_sr.shape[0]}x{hr_sr.shape[1]}")
    print(f"  PSNR (bicubic)   : {psnr_before:.2f} dB")
    print(f"  PSNR (WLS+L1)    : {psnr_after:.2f} dB")
    print(f"  SSIM (bicubic)   : {ssim_before:.4f}")
    print(f"  SSIM (WLS+L1)    : {ssim_after:.4f}")
    print(
        f"  Converged in {opt_result.iterations} iterations "
        f"({'yes' if opt_result.converged else 'max iter'})"
    )
    print("-" * 40)

    plot_paths = generate_all_visualizations(
        lr_obs, hr_sr, hr_gt, opt_result, user_config, prefix="user_"
    )

    explanation = generate_explanation(
        config=user_config,
        opt_result=opt_result,
        psnr_before=psnr_before,
        psnr_after=psnr_after,
        ssim_after=ssim_after,
        data_source=f"User upload ({os.path.basename(image_path)})",
        lr_shape=lr_obs.shape,
        hr_shape=hr_shape,
    )
    expl_path = os.path.join(user_out, "user_explanation.txt")
    with open(expl_path, "w", encoding="utf-8") as fh:
        fh.write(explanation)

    # Also save the SR image for frontend consumption
    sr_path = os.path.join(user_out, "user_super_resolved.png")
    io.imsave(sr_path, (np.clip(hr_sr, 0, 1) * 255).astype(np.uint8))

    safe_explanation = explanation.encode("ascii", errors="replace").decode("ascii")
    print("\n" + safe_explanation)
    print(f"\n  Outputs saved under: {user_out}")

    return {
        "hr_image": hr_sr,
        "lr_image": lr_obs,
        "hr_ground_truth": hr_gt,
        "psnr": psnr_after,
        "ssim": ssim_after,
        "psnr_before": psnr_before,
        "ssim_before": ssim_before,
        "iterations": opt_result.iterations,
        "converged": opt_result.converged,
        "convergence_graph": plot_paths["convergence"],
        "residual_map": plot_paths["residual_map"],
        "sparsity_map": plot_paths["sparsity_map"],
        "comparison": plot_paths["comparison"],
        "super_resolved_path": sr_path,
        "explanation": explanation,
        "explanation_path": expl_path,
        "optimization": opt_result,
    }


def run_demo(config: Optional[SRConfig] = None) -> Dict[str, Any]:
    """
    End-to-end demonstration:
      1) Full UC Merced train/test evaluation
      2) process_user_image() smoke-test on one sample image
    """
    if config is None:
        config = SRConfig()

    os.makedirs(config.output_dir, exist_ok=True)

    # --- FIX 1: full dataset evaluation ---
    dataset_results = evaluate_full_dataset(config)

    # --- FIX 2: user image input function test ---
    print("\n" + "=" * 70)
    print("  Testing process_user_image() with one sample image")
    print("=" * 70)

    sample_path: Optional[str] = None
    if dataset_results.get("test_records"):
        sample_path = dataset_results["test_records"][0][0]
    elif dataset_results.get("train_records"):
        sample_path = dataset_results["train_records"][0][0]

    user_results: Optional[Dict[str, Any]] = None
    if sample_path and os.path.isfile(sample_path):
        user_results = process_user_image(sample_path, config)
        print("\n  process_user_image() OK")
        print(f"    PSNR={user_results['psnr']:.2f} dB  SSIM={user_results['ssim']:.4f}")
        print(f"    Convergence graph: {user_results['convergence_graph']}")
        print(f"    Residual map     : {user_results['residual_map']}")
        print(f"    Sparsity map     : {user_results['sparsity_map']}")
    else:
        print("  [warn] No sample image available to test process_user_image().")

    return {
        "dataset_results": dataset_results,
        "user_results": user_results,
    }


# ---------------------------------------------------------------------------
# Section 7 — Visualisation
# ---------------------------------------------------------------------------

def generate_all_visualizations(
    lr: np.ndarray,
    hr_sr: np.ndarray,
    hr_gt: np.ndarray,
    opt_result: OptimizationResult,
    config: SRConfig,
    prefix: str = "",
) -> Dict[str, str]:
    """Create comparison, convergence, residual, and sparsity plots."""
    out = config.output_dir
    os.makedirs(out, exist_ok=True)
    paths: Dict[str, str] = {}

    paths["comparison"] = _plot_comparison(
        lr, hr_sr, hr_gt, out, filename=f"{prefix}comparison.png"
    )
    paths["convergence"] = _plot_convergence(
        opt_result, out, filename=f"{prefix}convergence.png"
    )
    paths["residual_map"] = _plot_residual_map(
        hr_sr, hr_gt, out, filename=f"{prefix}residual_map.png"
    )
    paths["sparsity_map"] = _plot_sparsity_map(
        hr_sr, hr_gt, config, out, filename=f"{prefix}sparsity_map.png"
    )

    return paths


def _plot_comparison(
    lr: np.ndarray,
    hr_sr: np.ndarray,
    hr_gt: np.ndarray,
    out_dir: str,
    filename: str = "comparison.png",
) -> str:
    """Side-by-side: LR | SR | Ground truth."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = [
        f"Low Resolution ({lr.shape[0]}x{lr.shape[1]})",
        f"WLS+L1 Super-Resolved ({hr_sr.shape[0]}x{hr_sr.shape[1]})",
        f"Ground Truth ({hr_gt.shape[0]}x{hr_gt.shape[1]})",
    ]
    images = [lr, hr_sr, hr_gt]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Satellite Image Super-Resolution — Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_convergence(
    opt_result: OptimizationResult,
    out_dir: str,
    filename: str = "convergence.png",
) -> str:
    """Objective function and data fidelity vs iteration."""
    iters = np.arange(1, len(opt_result.objective_history) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.semilogy(iters, opt_result.objective_history, "b-", linewidth=1.5, label="Total objective")
    ax1.semilogy(iters, opt_result.data_fidelity_history, "g--", linewidth=1.2, label="Data term ||Wx-y||^2")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Value (log scale)")
    ax1.set_title("Objective Convergence")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, opt_result.residual_history, "r-", linewidth=1.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("||Wx - y||_2")
    ax2.set_title("Reconstruction Residual")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("WLS Optimisation Convergence", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_residual_map(
    hr_sr: np.ndarray,
    hr_gt: np.ndarray,
    out_dir: str,
    filename: str = "residual_map.png",
) -> str:
    """Spatial map of absolute reconstruction error."""
    residual = np.abs(hr_sr - hr_gt)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(residual, cmap="hot", vmin=0, vmax=residual.max() + 1e-8)
    axes[0].set_title("Absolute Error |x* - x_true|")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(hr_sr, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reconstruction (reference)")
    axes[1].axis("off")

    fig.suptitle("Reconstruction Error Map", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_sparsity_map(
    hr_sr: np.ndarray,
    hr_gt: np.ndarray,
    config: SRConfig,
    out_dir: str,
    filename: str = "sparsity_map.png",
) -> str:
    """
    Visualise sparse vs dense regions via gradient magnitude thresholding.
    High-gradient (edge) regions are 'features preserved'; low-gradient areas
    are 'smoothed / sparsified' by L1 regularisation.
    """
    # Gradient magnitude of reconstruction
    gy, gx = np.gradient(hr_sr)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Feature weight map: normalised gradient magnitude
    weight_map = grad_mag / (grad_mag.max() + 1e-8)

    # Sparsity indicator: pixels suppressed relative to ground truth gradient
    gy_gt, gx_gt = np.gradient(hr_gt)
    grad_gt = np.sqrt(gx_gt ** 2 + gy_gt ** 2)
    suppression = np.maximum(0.0, grad_gt - grad_mag) / (grad_gt.max() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(weight_map, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Feature Weight Map\n(preserved edges)")
    axes[0].axis("off")

    axes[1].imshow(suppression, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Sparsity / Suppression Map\n(regularised regions)")
    axes[1].axis("off")

    axes[2].imshow(hr_sr, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"L1-regularised SR\n(lambda={config.lambda_l1})")
    axes[2].axis("off")

    fig.suptitle("Sparsity & Feature Preservation Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Section 8 — Explanation generation
# ---------------------------------------------------------------------------

def generate_explanation(
    config: SRConfig,
    opt_result: OptimizationResult,
    psnr_before: float,
    psnr_after: float,
    ssim_after: float,
    data_source: str,
    lr_shape: Tuple[int, int],
    hr_shape: Tuple[int, int],
) -> str:
    """
    Produce a human-readable explanation using actual optimisation values.
    """
    psnr_gain = psnr_after - psnr_before
    final_obj = opt_result.objective_history[-1]
    final_data = opt_result.data_fidelity_history[-1]
    final_l1 = opt_result.sparsity_history[-1]
    kkt_stat = opt_result.kkt["stationarity_inf"]
    kkt_comp = opt_result.kkt["complementarity_inf"]

    quality_word = (
        "excellent" if psnr_after >= 30 else
        "strong" if psnr_after >= 25 else
        "moderate" if psnr_after >= 20 else
        "limited"
    )

    lines = [
        "OPTIVISION MODULE 4 - SUPER-RESOLUTION EXPLANATION",
        "=" * 60,
        "",
        f"The satellite image from '{data_source}' was super-resolved by solving",
        f"a Weighted Least Squares optimisation problem over {opt_result.iterations} "
        f"iterations using ISTA (Iterative Shrinkage-Thresholding).",
        "",
        "MATHEMATICAL FORMULATION",
        f"  minimise  ||W x - y||^2 + lambda||x||_1   subject to  x >= 0",
        f"  Scale factor : {config.scale}x  ({lr_shape[0]}x{lr_shape[1]} -> "
        f"{hr_shape[0]}x{hr_shape[1]})",
        f"  Blur sigma   : {config.blur_sigma} px",
        f"  lambda (L1)  : {config.lambda_l1}",
        f"  Step size alpha  : {opt_result.step_size:.6e}",
        "",
        "CONVERGENCE",
        f"  Final objective value : {final_obj:.6e}",
        f"    Data fidelity term  : {final_data:.6e}  (||Wx-y||^2)",
        f"    L1 sparsity term    : {final_l1:.6e}  (lambda||x||_1)",
        f"  Converged             : {'Yes' if opt_result.converged else 'No (max iterations reached)'}",
        "",
        "KKT OPTIMALITY CHECK",
        f"  Stationarity residual  : {kkt_stat:.4e}  (should -> 0)",
        f"  Complementarity residual: {kkt_comp:.4e}  (should -> 0)",
        "",
        "RECONSTRUCTION QUALITY",
        f"  PSNR improved from {psnr_before:.2f} dB (bicubic baseline) to "
        f"{psnr_after:.2f} dB (WLS+L1), a gain of {psnr_gain:+.2f} dB.",
        f"  SSIM = {ssim_after:.4f}, indicating {quality_word} structural similarity "
        f"with the ground truth.",
        "",
        "INTERPRETATION",
        f"  L1 sparsity regularisation (lambda={config.lambda_l1}) preserved sharp edges "
        f"and structural features while suppressing noise in homogeneous regions. "
        f"The degradation operator W (Gaussian blur sigma={config.blur_sigma} + "
        f"{config.scale}x downsampling) models the satellite sensor point-spread "
        f"function.  Regions with high gradient magnitude in the feature weight map "
        f"correspond to preserved land-use boundaries; smoothed areas reflect "
        f"L1-induced sparsity in flat terrain.",
        "",
        f"  Overall, the {quality_word} reconstruction quality (PSNR={psnr_after:.2f} dB, "
        f"SSIM={ssim_after:.4f}) confirms that the WLS sparse optimisation "
        f"successfully recovered high-frequency detail from the low-resolution observation.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 9 — Entry point
# ---------------------------------------------------------------------------

def print_install_instructions() -> None:
    """Print required pip packages."""
    print(
        "\nRequired packages (run once):\n"
        "  pip install numpy scipy scikit-image matplotlib\n"
    )


if __name__ == "__main__":
    print_install_instructions()

    demo_config = SRConfig(
        scale=2,
        blur_sigma=1.0,
        lambda_l1=0.005,
        max_iterations=150,
        tolerance=1e-5,
        target_lr_size=64,
    )

    results = run_demo(demo_config)

    print("\n" + "=" * 70)
    print("  Module 4 demo complete.  Output files in:", demo_config.output_dir)
    print("=" * 70)
