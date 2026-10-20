"""Shared synthetic tables for unit tests (not CIC-IDS2017)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from module1.data.loader import make_synthetic_flows


def tiny_csv(path: Path) -> Path:
    loaded = make_synthetic_flows(n_samples=80, n_features=6, seed=0)
    loaded.frame.to_csv(path, index=False)
    return path


def csv_with_junk(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            " Flow Duration": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            " Flow Bytes/s": [1.0, float("inf"), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "const": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            " Label": [
                "BENIGN",
                "BENIGN",
                "DoS Hulk",
                "BENIGN",
                "PortScan",
                "BENIGN",
                "DDoS",
                "BENIGN",
                "BENIGN",
                "DoS Hulk",
                "PortScan",
                "BENIGN",
            ],
        }
    )
    frame.to_csv(path, index=False)
    return path
