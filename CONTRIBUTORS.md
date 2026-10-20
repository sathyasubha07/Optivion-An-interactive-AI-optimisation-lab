# Contributors — OptiVision (Team 08)

OptiVision is a group project for the MFC course, built by Team 08 as a whole.

| Part | Method / stack |
|---|---|
| Module 1 — Network Intrusion Detection | Soft-margin linear SVM + L1 sparsity, convex QP solved with OSQP |
| Module 4 — Satellite Super-Resolution | Weighted least squares + L1 sparsity, solved with ISTA |
| Backend / frontend integration | FastAPI registry + adapters, React workspace |

The integration layer (`backend/` and `frontend/`) calls each module through an adapter and never modifies
module source files.
