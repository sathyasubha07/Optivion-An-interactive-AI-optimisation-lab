// Deterministic accent per module id, so new modules get a look without frontend changes.
export function moduleHue(id = '') {
  let h = 0
  for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) % 360
  return h
}
