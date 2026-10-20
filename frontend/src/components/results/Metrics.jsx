import { formatDelta, formatValue, isMissing } from '../../lib/format'
import { Glass } from '../ui'

function Value({ metric, className = '' }) {
  const missing = isMissing(metric.value)
  const text = formatValue(metric.value, metric.fmt)
  if (missing) return <span className={`text-mist-500 italic ${className}`}>not computed</span>
  const boolTone =
    metric.fmt === 'bool' ? (metric.value === true ? 'text-teal' : metric.value === false ? 'text-amber' : '') : ''
  return (
    <span className={`${boolTone} ${className}`}>
      {text}
      {metric.unit && <span className="ml-1 text-[0.6em] font-normal text-mist-500">{metric.unit}</span>}
    </span>
  )
}

function Delta({ metric }) {
  if (!metric.baseline_label) return null
  const delta = formatDelta(metric.value, metric.baseline, metric.fmt)
  if (!delta) return <p className="mt-2 text-xs text-mist-500">{metric.baseline_label}: not computed</p>
  const better =
    metric.higher_is_better === undefined
      ? null
      : metric.higher_is_better
        ? metric.value >= metric.baseline
        : metric.value <= metric.baseline
  const tone = better === null ? 'text-mist-300' : better ? 'text-teal' : 'text-amber'
  return (
    <p className="mt-2 text-xs text-mist-500">
      <span className={`font-mono ${tone}`}>
        {delta}
        {metric.unit ? ` ${metric.unit}` : ''}
      </span>{' '}
      vs {metric.baseline_label} ({formatValue(metric.baseline, metric.fmt)})
    </p>
  )
}

export function SummaryCards({ metrics = [] }) {
  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {metrics.map((m) => (
        <Glass key={m.key} className="p-5">
          <p className="text-xs uppercase tracking-wider text-mist-500">{m.label}</p>
          <p className="mt-2 font-display text-2xl font-semibold text-mist-100 sm:text-3xl">
            <Value metric={m} />
          </p>
          <Delta metric={m} />
          {m.hint && <p className="mt-2 text-xs leading-snug text-mist-500">{m.hint}</p>}
        </Glass>
      ))}
    </div>
  )
}

export function MetricGroups({ groups = [] }) {
  return (
    <div className="grid gap-5 md:grid-cols-2">
      {groups.map((g) => (
        <Glass key={g.title} className="p-5">
          <h4 className="font-display font-semibold text-mist-100">{g.title}</h4>
          {g.note && <p className="mt-1 text-xs leading-relaxed text-mist-500">{g.note}</p>}
          <dl className="mt-4 divide-y divide-white/5">
            {g.metrics.map((m) => (
              <div key={m.key} className="flex items-baseline justify-between gap-4 py-2 text-sm">
                <dt className="text-mist-300">{m.label}</dt>
                <dd className="text-right font-mono text-mist-100">
                  <Value metric={m} />
                </dd>
              </div>
            ))}
          </dl>
        </Glass>
      ))}
    </div>
  )
}
