# OptiVision frontend

React + Vite + Tailwind CSS 4 + Framer Motion + Recharts + KaTeX.

```powershell
npm install; npm run dev
```

Runs on http://localhost:5173 and proxies `/api` and `/outputs` to the backend at http://127.0.0.1:8000.
Set `OPTIVISION_BACKEND` to point the proxy elsewhere.

Everything module-specific comes from the backend (`GET /modules` and the job result), so adding a module
needs no change here. See the result-shape contract in `../backend/adapters/base.py`.

- `src/pages/` — Landing, Modules, Workspace (the 7-step stepper), About
- `src/components/results/` — generic renderers for metrics, replay, images, charts, tables, downloads
- `src/lib/format.js` — value formatting. Missing values render as "not computed"; nothing is derived.
