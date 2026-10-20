"""Command-line entry for train / serve / experiment.

Examples:

    python -m module1.cli train --source synthetic
    python -m module1.cli train --source cicids2017
    python -m module1.cli experiment --source synthetic
    python -m module1.cli serve
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from module1.config.settings import load_config
from module1.logging_setup import configure_logging
from module1.services.nid_service import NIDService


def _save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OPTIVION Module 1 backend CLI")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the SVM+L1 detector")
    p_train.add_argument("--source", choices=["cicids2017", "synthetic"], default="cicids2017")
    p_train.add_argument("--out", default="artifacts/last_train.json")

    p_exp = sub.add_parser("experiment", help="Baseline and small C/λ grid")
    p_exp.add_argument("--source", choices=["cicids2017", "synthetic"], default="synthetic")
    p_exp.add_argument("--out", default="artifacts/last_experiment.json")

    p_serve = sub.add_parser("serve", help="Start the FastAPI backend")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    configure_logging(cfg.logging.level)
    svc = NIDService(cfg)

    if args.command == "train":
        result = svc.train(dataset_source=args.source)
        payload = result.model_dump()
        _save(payload, Path(args.out))
        ev = result.evaluation
        print(f"protocol={result.dataset.evaluation_protocol} source={result.dataset.source}")
        print(f"accuracy={ev.accuracy:.4f} attack_recall={ev.attack_recall:.4f} f1={ev.f1:.4f}")
        print(f"sparsity={result.sparsity.sparsity_percentage:.2f}% objective={result.optimization.primal_objective}")
        print(f"wrote {args.out}")
        return

    if args.command == "experiment":
        payload = svc.experiment(dataset_source=args.source)
        _save(payload, Path(args.out))
        print(json.dumps({"primary": payload["primary"], "baseline_lambda_0": payload["baseline_lambda_0"]}, indent=2))
        print(f"wrote {args.out}")
        return

    if args.command == "serve":
        import uvicorn

        from module1.api.app import app

        host = args.host or cfg.api.host
        port = args.port or cfg.api.port
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
