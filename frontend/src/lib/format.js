// Formatting only. Never derives or invents values: if the backend did not return a
// finite number, the UI shows "not computed".
export const NOT_COMPUTED = 'not computed'

export const isMissing = (v) =>
  v === null || v === undefined || v === NOT_COMPUTED || (typeof v === 'number' && !Number.isFinite(v))

function sci(v) {
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a >= 1e-3 && a < 1e5) return Number(v.toPrecision(5)).toString()
  return v.toExponential(3)
}

export function formatValue(value, fmt = 'float4') {
  if (isMissing(value)) return NOT_COMPUTED
  switch (fmt) {
    case 'bool':
      return typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value)
    case 'text':
      return String(value)
    case 'int':
      return typeof value === 'number' ? Math.round(value).toLocaleString() : String(value)
    case 'pct':
      return typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : String(value)
    case 'float2':
      return typeof value === 'number' ? value.toFixed(2) : String(value)
    case 'sci':
      return typeof value === 'number' ? sci(value) : String(value)
    case 'float4':
    default:
      return typeof value === 'number' ? value.toFixed(4) : String(value)
  }
}

// Difference between two backend-returned numbers, shown next to a baseline.
export function formatDelta(value, baseline, fmt) {
  if (isMissing(value) || isMissing(baseline) || typeof value !== 'number' || typeof baseline !== 'number') return null
  const d = value - baseline
  const sign = d >= 0 ? '+' : '−'
  const abs = Math.abs(d)
  if (fmt === 'pct') return `${sign}${(abs * 100).toFixed(2)} pp`
  if (fmt === 'float2') return `${sign}${abs.toFixed(2)}`
  return `${sign}${abs.toFixed(4)}`
}

export function seriesIsUsable(values) {
  return Array.isArray(values) && values.some((v) => typeof v === 'number')
}
