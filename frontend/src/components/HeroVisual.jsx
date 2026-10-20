import { motion } from 'framer-motion'

// Decorative illustration of an iterative method descending a convex bowl.
// It is not data and shows no numbers.
const path = 'M 70 70 L 330 120 L 120 210 L 290 150 L 175 195 L 245 172 L 205 186 L 222 180'

export default function HeroVisual() {
  return (
    <div className="relative mx-auto aspect-[4/3] w-full max-w-[520px]" aria-hidden="true">
      <div className="glass absolute inset-0 rounded-3xl" />
      <svg viewBox="0 0 400 300" className="relative h-full w-full">
        <defs>
          <linearGradient id="hv-g" x1="0" x2="1">
            <stop offset="0" stopColor="#5eead4" />
            <stop offset="1" stopColor="#a78bfa" />
          </linearGradient>
          <radialGradient id="hv-r">
            <stop offset="0" stopColor="#5eead4" stopOpacity=".45" />
            <stop offset="1" stopColor="#5eead4" stopOpacity="0" />
          </radialGradient>
        </defs>
        {[150, 118, 90, 64, 40, 20].map((rx, i) => (
          <motion.ellipse
            key={rx}
            cx="215"
            cy="180"
            rx={rx}
            ry={rx * 0.55}
            fill="none"
            stroke="url(#hv-g)"
            strokeWidth="1.2"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 0.18 + i * 0.1, scale: 1 }}
            transition={{ delay: 0.1 * i, duration: 0.8 }}
            style={{ transformOrigin: '215px 180px' }}
          />
        ))}
        <circle cx="222" cy="180" r="40" fill="url(#hv-r)" />
        <motion.path
          d={path}
          fill="none"
          stroke="url(#hv-g)"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 3.2, ease: 'easeInOut', repeat: Infinity, repeatDelay: 1.4 }}
        />
        {path
          .split(/[ML]/)
          .filter(Boolean)
          .map((pt, i) => {
            const [x, y] = pt.trim().split(' ').map(Number)
            return (
              <motion.circle
                key={i}
                cx={x}
                cy={y}
                r={i === 7 ? 5 : 3}
                fill={i === 7 ? '#5eead4' : '#eef2fb'}
                initial={{ opacity: 0 }}
                animate={{ opacity: [0, 1, 1, 0] }}
                transition={{ duration: 4.6, times: [0, 0.08, 0.85, 1], delay: (i / 8) * 3.2, repeat: Infinity, repeatDelay: 0 }}
              />
            )
          })}
        <text x="24" y="30" fill="#7d88a6" fontFamily="JetBrains Mono, monospace" fontSize="11">
          x⁽ᵏ⁺¹⁾ = prox(xᵏ − α∇f(xᵏ))
        </text>
        <text x="24" y="282" fill="#7d88a6" fontFamily="JetBrains Mono, monospace" fontSize="10">
          illustration
        </text>
      </svg>
    </div>
  )
}
