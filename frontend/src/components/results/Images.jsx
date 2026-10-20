import { AnimatePresence, motion } from 'framer-motion'
import { useCallback, useRef, useState } from 'react'
import { Glass } from '../ui'

export function CompareSlider({ before, after, label }) {
  const [pos, setPos] = useState(50)
  const ref = useRef(null)
  const dragging = useRef(false)

  const update = useCallback((clientX) => {
    const rect = ref.current?.getBoundingClientRect()
    if (!rect) return
    setPos(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)))
  }, [])

  if (!before?.url || !after?.url) return null
  return (
    <Glass className="p-4 sm:p-5">
      <div
        ref={ref}
        className="relative mx-auto aspect-square w-full max-w-[520px] cursor-ew-resize touch-none select-none overflow-hidden rounded-xl bg-black"
        onPointerDown={(e) => {
          dragging.current = true
          e.currentTarget.setPointerCapture(e.pointerId)
          update(e.clientX)
        }}
        onPointerMove={(e) => dragging.current && update(e.clientX)}
        onPointerUp={() => (dragging.current = false)}
      >
        <img src={after.url} alt={after.label} className="absolute inset-0 h-full w-full object-cover" draggable="false" />
        <img
          src={before.url}
          alt={before.label}
          className={`absolute inset-0 h-full w-full object-cover ${before.pixelated ? 'pixelated' : ''}`}
          style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
          draggable="false"
        />
        <div className="absolute inset-y-0 w-0.5 bg-white/90 shadow-[0_0_12px] shadow-white/60" style={{ left: `${pos}%` }}>
          <div className="absolute top-1/2 left-1/2 grid h-9 w-9 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white text-ink-950 shadow-lg">
            ⇆
          </div>
        </div>
        <span className="absolute left-3 top-3 rounded-md bg-black/60 px-2 py-1 font-mono text-[11px]">{before.label}</span>
        <span className="absolute right-3 top-3 rounded-md bg-black/60 px-2 py-1 font-mono text-[11px]">{after.label}</span>
      </div>
      <input
        type="range"
        min="0"
        max="100"
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        className="scrubber mx-auto mt-4 block w-full max-w-[520px]"
        aria-label={label || 'Compare images'}
      />
      {label && <p className="mt-2 text-center text-xs text-mist-500">{label}</p>}
    </Glass>
  )
}

function Lightbox({ item, onClose }) {
  return (
    <AnimatePresence>
      {item && (
        <motion.div
          className="fixed inset-0 z-50 grid place-items-center bg-black/80 p-4 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          role="dialog"
          aria-label={item.label}
        >
          <motion.figure
            initial={{ scale: 0.95 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0.95 }}
            className="max-h-full max-w-5xl"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={item.url}
              alt={item.label}
              className={`max-h-[80vh] w-auto rounded-xl bg-white ${item.pixelated ? 'pixelated' : ''}`}
            />
            <figcaption className="mt-3 flex items-center justify-between gap-4 text-sm text-mist-300">
              <span>{item.label}</span>
              <button type="button" onClick={onClose} className="rounded-lg bg-white/10 px-3 py-1.5 text-mist-100">
                Close
              </button>
            </figcaption>
          </motion.figure>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export function ImageGrid({ items = [], columns = 'sm:grid-cols-3', light = false }) {
  const [open, setOpen] = useState(null)
  const visible = items.filter((i) => i.url)
  if (!visible.length) return null
  return (
    <>
      <div className={`grid gap-4 ${columns}`}>
        {visible.map((item) => (
          <Glass key={item.key} as="figure" className="overflow-hidden p-3">
            <button type="button" className="block w-full" onClick={() => setOpen(item)} aria-label={`Enlarge ${item.label}`}>
              <img
                src={item.url}
                alt={item.label}
                loading="lazy"
                className={`w-full rounded-lg ${light ? 'bg-white' : 'aspect-square bg-black object-cover'} ${item.pixelated ? 'pixelated' : ''}`}
              />
            </button>
            <figcaption className="px-1 pt-3">
              <p className="text-sm font-medium text-mist-100">{item.label}</p>
              {item.caption && <p className="mt-1 text-xs leading-relaxed text-mist-500">{item.caption}</p>}
            </figcaption>
          </Glass>
        ))}
      </div>
      <Lightbox item={open} onClose={() => setOpen(null)} />
    </>
  )
}
