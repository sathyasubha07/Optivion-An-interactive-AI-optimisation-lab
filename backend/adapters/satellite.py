"""Adapter for Module 4 — calls process_user_image() only. Module source is not modified."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

from backend.adapters.base import Adapter, RunContext, Sample, metric
from backend.jsonutil import (
    NOT_COMPUTED,
    download,
    float_list,
    maybe_bool,
    maybe_float,
    maybe_int,
    path_to_url,
    save_gray_png,
)
from backend.paths import MODULE4_ROOT

UC_MERCED_IMAGES = MODULE4_ROOT / "data" / "UCMerced_LandUse" / "Images"
# A few visually distinct scenes from the locally downloaded dataset (gitignored; may be absent).
UC_MERCED_PICKS = [
    ("airplane", "airplane00.tif"),
    ("harbor", "harbor10.tif"),
    ("denseresidential", "denseresidential05.tif"),
    ("freeway", "freeway03.tif"),
]


def _load_module4():
    import matplotlib

    matplotlib.use("Agg")
    module_path = MODULE4_ROOT / "Module4_Satellite_SuperResolution.py"
    if not module_path.is_file():
        raise RuntimeError(f"Module 4 source not found: {module_path}")
    name = "module4_satellite_super_resolution"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Module 4.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def list_samples() -> list[Sample]:
    samples = []
    for cls, filename in UC_MERCED_PICKS:
        if (UC_MERCED_IMAGES / cls / filename).is_file():
            samples.append(
                Sample(
                    id=f"ucm-{cls}",
                    label=f"UC Merced: {cls}",
                    description=f"{filename} from the locally downloaded UC Merced Land Use dataset.",
                )
            )
    samples.append(
        Sample(
            id="synthetic",
            label="Synthetic scene",
            description="Satellite-like image from Module 4's own fallback generator (fields, road, buildings).",
        )
    )
    return samples


def make_sample(sample_id: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for cls, filename in UC_MERCED_PICKS:
        if sample_id == f"ucm-{cls}":
            source = UC_MERCED_IMAGES / cls / filename
            if not source.is_file():
                raise FileNotFoundError(f"UC Merced sample not available locally: {source}")
            return Path(shutil.copy2(source, dest_dir / filename))
    if sample_id == "synthetic":
        m4 = _load_module4()
        return save_gray_png(m4._generate_synthetic_satellite_image(128), dest_dir / "synthetic_scene.png")
    raise ValueError(f"Unknown sample '{sample_id}'.")


def run(ctx: RunContext) -> dict[str, Any]:
    m4 = _load_module4()
    config = m4.SRConfig(output_dir=str(ctx.out_dir))
    raw = m4.process_user_image(str(ctx.input_path), config)

    opt = raw.get("optimization")
    kkt = (getattr(opt, "kkt", None) or {}) if opt is not None else {}
    history = {
        "objective": float_list(getattr(opt, "objective_history", None)),
        "data_term": float_list(getattr(opt, "data_fidelity_history", None)),
        "sparsity_term": float_list(getattr(opt, "sparsity_history", None)),
        "residual": float_list(getattr(opt, "residual_history", None)),
    }
    final_objective = (
        history["objective"][-1] if isinstance(history["objective"], list) else NOT_COMPUTED
    )

    lr_png = save_gray_png(raw["lr_image"], ctx.out_dir / "lr_observation.png")
    gt_png = save_gray_png(raw["hr_ground_truth"], ctx.out_dir / "ground_truth.png")
    sr_png = raw.get("super_resolved_path")
    lr_h, lr_w = raw["lr_image"].shape
    hr_h, hr_w = raw["hr_image"].shape

    psnr = maybe_float(raw.get("psnr"))
    ssim = maybe_float(raw.get("ssim"))
    psnr_before = maybe_float(raw.get("psnr_before"))
    ssim_before = maybe_float(raw.get("ssim_before"))
    iterations = maybe_int(raw.get("iterations"))
    converged = maybe_bool(raw.get("converged"))

    images = [
        {"key": "lr", "label": f"Low-res input ({lr_w}×{lr_h})", "url": path_to_url(lr_png), "pixelated": True,
         "caption": "Synthesised observation y = downsample(blur(x_gt)). Shown enlarged with nearest-neighbour pixels."},
        {"key": "sr", "label": f"WLS + L1 reconstruction ({hr_w}×{hr_h})", "url": path_to_url(sr_png),
         "caption": "Final ISTA iterate x returned by the module."},
        {"key": "gt", "label": f"Ground truth ({hr_w}×{hr_h})", "url": path_to_url(gt_png),
         "caption": "Your image after the module's grayscale + resize preprocessing."},
    ]

    return {
        "summary": [
            metric("psnr", "PSNR", psnr, "float2", unit="dB", baseline=psnr_before,
                   baseline_label="bicubic", higher_is_better=True),
            metric("ssim", "SSIM", ssim, "float4", baseline=ssim_before,
                   baseline_label="bicubic", higher_is_better=True),
            metric("iterations", "ISTA iterations", iterations, "int"),
            metric("converged", "Converged", converged, "bool",
                   hint=f"Relative objective change below tolerance {config.tolerance:g}."),
        ],
        "metric_groups": [
            {
                "title": "Reconstruction quality vs ground truth",
                "metrics": [
                    metric("psnr", "PSNR (WLS + L1)", psnr, "float2", unit="dB"),
                    metric("psnr_before", "PSNR (bicubic baseline)", psnr_before, "float2", unit="dB"),
                    metric("ssim", "SSIM (WLS + L1)", ssim, "float4"),
                    metric("ssim_before", "SSIM (bicubic baseline)", ssim_before, "float4"),
                ],
            },
            {
                "title": "Optimisation (ISTA)",
                "metrics": [
                    metric("iterations", "Iterations", iterations, "int"),
                    metric("max_iterations", "Iteration cap", maybe_int(config.max_iterations), "int"),
                    metric("converged", "Converged", converged, "bool"),
                    metric("final_objective", "Final objective", final_objective, "sci"),
                    metric("step_size", "Step size α = 1/L", maybe_float(getattr(opt, "step_size", None)), "sci"),
                    metric("lambda_l1", "λ (L1 weight)", maybe_float(config.lambda_l1), "sci"),
                    metric("tolerance", "Tolerance", maybe_float(config.tolerance), "sci"),
                ],
            },
            {
                "title": "KKT optimality residuals",
                "note": "Infinity-norm residuals computed by the module's compute_kkt_residuals(). Closer to 0 is closer to optimal.",
                "metrics": [
                    metric("stationarity_inf", "Stationarity ‖·‖∞", maybe_float(kkt.get("stationarity_inf")), "sci"),
                    metric("complementarity_inf", "Complementarity ‖ν∘x‖∞", maybe_float(kkt.get("complementarity_inf")), "sci"),
                    metric("dual_norm_inf", "Dual ‖ν‖∞", maybe_float(kkt.get("dual_norm_inf")), "sci"),
                ],
            },
        ],
        "replay": {
            "available": isinstance(history["objective"], list),
            "note": "Replay of the iterations recorded by the solver during this run (not a live stream).",
            "x_label": "ISTA iteration",
            "series": [
                {"key": "objective", "label": "Objective ‖Wx−y‖² + λ‖x‖₁", "values": history["objective"], "log_scale": True},
                {"key": "data_term", "label": "Data term ‖Wx−y‖²", "values": history["data_term"], "log_scale": True},
                {"key": "sparsity_term", "label": "L1 term λ‖x‖₁", "values": history["sparsity_term"]},
                {"key": "residual", "label": "Residual ‖Wx−y‖₂", "values": history["residual"], "log_scale": True},
            ],
        },
        "images": images,
        "compare": {"before": "lr", "after": "sr", "label": "Drag to compare the low-res input with the reconstruction"},
        "plots": [
            {"key": "comparison", "label": "Side-by-side comparison", "url": path_to_url(raw.get("comparison"))},
            {"key": "convergence", "label": "Convergence curves", "url": path_to_url(raw.get("convergence_graph"))},
            {"key": "residual_map", "label": "Residual map |x − x_gt|", "url": path_to_url(raw.get("residual_map"))},
            {"key": "sparsity_map", "label": "Sparsity and feature preservation", "url": path_to_url(raw.get("sparsity_map"))},
        ],
        "explanation": raw.get("explanation") or NOT_COMPUTED,
        "explanation_source": "Written by Module 4's generate_explanation() for this run.",
        "notes": [
            "Your upload is treated as ground truth. Module 4 converts it to grayscale, resizes it to "
            f"{hr_w}×{hr_h}, synthesises a {lr_w}×{lr_h} low-resolution observation by blur + downsample, "
            "then reconstructs. PSNR/SSIM compare the reconstruction with that ground truth.",
            "The bicubic baseline is the module's own bicubic upsample of the same low-res observation.",
        ],
        "downloads": [
            d for d in (
                download("Super-resolved image (PNG)", sr_png),
                download("Low-res observation (PNG)", lr_png),
                download("Ground truth (PNG)", gt_png),
                download("Explanation (TXT)", raw.get("explanation_path")),
                download("Convergence plot (PNG)", raw.get("convergence_graph")),
            ) if d
        ],
    }


ADAPTER = Adapter(run=run, list_samples=list_samples, make_sample=make_sample)
