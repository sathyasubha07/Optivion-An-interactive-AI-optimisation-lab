# OPTIVION — An Interactive AI Optimisation Lab

OPTIVION is a multi-module project exploring how different optimisation techniques can be applied to real-world problems. Each module tackles a separate domain using a distinct mathematical optimisation approach, implemented and evaluated independently.

This repository is organised as **one module per branch**, so each module can be developed, tested, and reviewed on its own without affecting the others.

---

## Modules

| Module | Domain | Status | Branch |
|---|---|---|---|
| Module 1 | Network Intrusion Detection (SVM + L1 sparsity, convex QP, OSQP) | Implemented, tested, evaluated on real CIC-IDS2017 data | `Module1_Network_Intrusion_Detection` |
| Module 2 | Satellite Image Super-Resolution | Implemented, tested, evaluated | `Module4_Satellite_SuperResolution` |
| Future modules | — | Not yet started | — |

Each module's branch contains its own README with full details on that module's method, dataset, results, and how to run it.

---

## Project structure

```
Optivion-An-interactive-AI-optimisation-lab/
├── Module1_Network_Intrusion_Detection/   # SVM-based intrusion detection (this branch)
├── Module4_Satellite_SuperResolution/     # Satellite image super-resolution (separate branch)
└── ...future modules                      # Not yet started
```

Modules are kept isolated from each other by design — no shared code, no shared dependencies — so each one can be built, tested, and run independently.

---

## Status

This project is under active development as part of academic coursework. Not all modules described above are complete; see each branch's own README for the accurate, current status of that specific module.

---

## License

See `LICENSE` for details.
