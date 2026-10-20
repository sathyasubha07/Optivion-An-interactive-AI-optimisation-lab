// Thin client for the OptiVision backend. All module knowledge comes from the API.
const BASE = import.meta.env.VITE_API_BASE ?? '/api'

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function request(path, options) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, options)
  } catch {
    throw new ApiError('Cannot reach the OptiVision backend. Is uvicorn running on port 8000?', 0)
  }
  let body = null
  try {
    body = await res.json()
  } catch {
    // non-JSON (e.g. proxy error page)
  }
  if (!res.ok) {
    const detail = body?.detail
    const message =
      typeof detail === 'string'
        ? detail
        : res.status >= 500 && !body
          ? 'Cannot reach the OptiVision backend. Is uvicorn running on port 8000?'
          : `Request failed (${res.status}).`
    throw new ApiError(message, res.status)
  }
  return body
}

export const fetchModules = () => request('/modules').then((d) => d.modules)
export const fetchModule = (id) => request(`/modules/${encodeURIComponent(id)}`)
export const fetchContributors = () => request('/contributors')
export const fetchJob = (jobId) => request(`/jobs/${encodeURIComponent(jobId)}`)

export function startRun(moduleId, { file, sampleId }) {
  const form = new FormData()
  if (file) form.append('file', file)
  if (sampleId) form.append('sample_id', sampleId)
  return request(`/modules/${encodeURIComponent(moduleId)}/run`, { method: 'POST', body: form })
}

// Static outputs are served by the backend under /outputs (proxied in dev).
export const assetUrl = (url) => url ?? null
