import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { moduleHue } from '../lib/hue'

export function Glass({ as: Tag = 'div', className = '', children, ...rest }) {
  return (
    <Tag className={`glass rounded-2xl ${className}`} {...rest}>
      {children}
    </Tag>
  )
}

export function Chip({ children, tone = 'default', className = '' }) {
  const tones = {
    default: 'border-white/10 bg-white/5 text-mist-300',
    teal: 'border-teal/30 bg-teal/10 text-teal',
    violet: 'border-violet/30 bg-violet/10 text-violet',
    amber: 'border-amber/30 bg-amber/10 text-amber',
    rose: 'border-rose/30 bg-rose/10 text-rose',
  }
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 font-mono text-[11px] tracking-wide ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  )
}

export function StatusPill({ status }) {
  if (status === 'ready')
    return (
      <Chip tone="teal">
        <span className="h-1.5 w-1.5 rounded-full bg-teal shadow-[0_0_8px] shadow-teal" />
        Ready
      </Chip>
    )
  return <Chip tone="amber">Coming soon</Chip>
}

export function Eyebrow({ children, className = '' }) {
  return <p className={`font-mono text-xs uppercase tracking-[0.2em] text-teal/80 ${className}`}>{children}</p>
}

export function SectionTitle({ eyebrow, title, children, className = '' }) {
  return (
    <div className={`max-w-2xl ${className}`}>
      {eyebrow && <Eyebrow className="mb-3">{eyebrow}</Eyebrow>}
      <h2 className="font-display text-3xl font-semibold tracking-tight text-mist-100 sm:text-4xl">{title}</h2>
      {children && <p className="mt-4 text-base leading-relaxed text-mist-300">{children}</p>}
    </div>
  )
}

const btnBase =
  'inline-flex items-center justify-center gap-2 rounded-xl px-5 py-2.5 text-sm font-semibold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-teal/60 disabled:cursor-not-allowed disabled:opacity-40'
const btnTones = {
  primary: 'bg-gradient-accent text-ink-950 shadow-[0_8px_30px_-8px] shadow-teal/50 hover:brightness-110',
  ghost: 'border border-white/10 bg-white/5 text-mist-100 hover:bg-white/10',
  subtle: 'text-mist-300 hover:text-mist-100',
}

export function Button({ tone = 'primary', to, href, className = '', children, ...rest }) {
  const cls = `${btnBase} ${btnTones[tone]} ${className}`
  if (to)
    return (
      <Link to={to} className={cls} {...rest}>
        {children}
      </Link>
    )
  if (href)
    return (
      <a href={href} className={cls} {...rest}>
        {children}
      </a>
    )
  return (
    <button type="button" className={cls} {...rest}>
      {children}
    </button>
  )
}

export function FadeIn({ children, delay = 0, className = '', y = 16 }) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.55, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  )
}

export function Spinner({ className = 'h-4 w-4' }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity=".2" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}

export function ErrorBox({ title = 'Something went wrong', children, action }) {
  return (
    <div className="rounded-2xl border border-rose/30 bg-rose/10 p-5 text-sm text-rose">
      <p className="font-semibold">{title}</p>
      <div className="mt-1 break-words text-rose/90">{children}</div>
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function ModuleGlyph({ module, size = 'h-12 w-12' }) {
  const hue = moduleHue(module.id)
  return (
    <div
      className={`${size} grid shrink-0 place-items-center rounded-xl border border-white/10 font-display text-lg font-semibold text-ink-950`}
      style={{
        background: `linear-gradient(135deg, hsl(${hue} 85% 72%), hsl(${(hue + 60) % 360} 80% 70%))`,
      }}
      aria-hidden="true"
    >
      {module.number ?? '•'}
    </div>
  )
}
