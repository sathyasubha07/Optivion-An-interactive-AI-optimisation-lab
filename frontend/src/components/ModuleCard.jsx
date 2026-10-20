import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { Chip, ModuleGlyph, StatusPill } from './ui'

export default function ModuleCard({ module, index = 0 }) {
  const ready = module.status === 'ready'
  const body = (
    <>
      <div className="flex items-start justify-between gap-3">
        <ModuleGlyph module={module} />
        <StatusPill status={module.status} />
      </div>
      <p className="mt-5 font-mono text-[11px] uppercase tracking-[0.18em] text-mist-500">
        Module {module.number} · {module.category}
      </p>
      <h3 className="mt-1.5 font-display text-xl font-semibold text-mist-100">{module.name}</h3>
      <p className="mt-2 flex-1 text-sm leading-relaxed text-mist-300">{module.tagline || module.short_description}</p>
      <div className="mt-4 flex flex-wrap gap-1.5">
        {module.method_chips?.map((c) => (
          <Chip key={c}>{c}</Chip>
        ))}
      </div>
      <div className="mt-5 flex items-center justify-between border-t border-white/5 pt-4 text-xs">
        <span className="text-mist-500">{module.input_type ? `Input: ${module.input_type}` : ''}</span>
        {ready ? (
          <span className="font-semibold text-teal transition group-hover:translate-x-0.5">Open lab →</span>
        ) : (
          <span className="text-mist-500">In development</span>
        )}
      </div>
    </>
  )

  const cls = 'glass group flex h-full flex-col rounded-2xl p-6 transition'
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.5, delay: index * 0.07 }}
      whileHover={ready ? { y: -4 } : undefined}
      className="h-full"
    >
      {ready ? (
        <Link
          to={`/modules/${module.id}`}
          className={`${cls} hover:border-teal/30 hover:shadow-[0_20px_60px_-20px] hover:shadow-teal/30`}
        >
          {body}
        </Link>
      ) : (
        <div className={`${cls} opacity-60`} aria-disabled="true">
          {body}
        </div>
      )}
    </motion.div>
  )
}
