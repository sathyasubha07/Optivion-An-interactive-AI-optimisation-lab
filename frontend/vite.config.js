import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The backend runs on :8000 (uvicorn backend.main:app). The dev server proxies to it,
// so the app only ever uses relative URLs (/api/..., /outputs/...).
const BACKEND = process.env.OPTIVISION_BACKEND ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, '') },
      '/outputs': { target: BACKEND, changeOrigin: true },
    },
  },
  preview: {
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, '') },
      '/outputs': { target: BACKEND, changeOrigin: true },
    },
  },
})
